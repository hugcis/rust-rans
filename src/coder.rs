const PROB_BITS: u32 = 14;
const RANS64_L: u64 = 1 << 31;

type RansState = u64;

fn find_in_int_dist(cdf: &[u32], to_find: u64) -> u64 {
    let mut r = 0;
    for i in 0..255 {
        if (cdf[i + 1] as u64 > to_find) && (cdf[i] as u64 <= to_find) {
            r = i as u64;
            break;
        }
    }
    r
}

enum Mode {
    Static,
    Precomp,
    Dyn,
}

#[derive(Default, Clone, Copy)]
struct PreCompSym {
    rcp_freq: u64,
    freq: u32,
    bias: u32,
    cmpl_freq: u32,
    rcp_shift: u32,
}

pub struct SymbolStats {
    pdf: [u32; 256],
    cdf: [u32; 257],
    probs: [u32; 256],
    model_mode: Mode,
    precomp: [PreCompSym; 256],
}

impl SymbolStats {
    /// Creates a new empty SymbolStats object. The probability distribution
    /// should be set before using.
    pub fn new() -> SymbolStats {
        SymbolStats {
            pdf: [0; 256],
            cdf: [0; 257],
            probs: [0; 256],
            model_mode: Mode::Dyn,
            precomp: [PreCompSym::default(); 256],
        }
    }

    /// Creates a new SymbolStats from a probability count. It needs not be
    /// normalized.
    pub fn new_static(probs: &[u32]) -> SymbolStats {
        let mut stats = SymbolStats {
            pdf: [0; 256],
            cdf: [0; 257],
            probs: [0; 256],
            model_mode: Mode::Static,
            precomp: [PreCompSym::default(); 256],
        };
        stats.update_probs(probs);
        stats
    }

    /// Creates a new SymbolStats from a probability count in pre-computed mode.
    /// It needs not be normalized.
    fn new_precomp(probs: &[u32]) -> SymbolStats {
        let mut stats = SymbolStats {
            pdf: [0; 256],
            cdf: [0; 257],
            probs: [0; 256],
            model_mode: Mode::Precomp,
            precomp: [PreCompSym::default(); 256],
        };
        stats.update_probs(probs);
        stats
    }

    fn precomp(&mut self) {
        for i in 0..256 {
            let s = &mut self.precomp[i];
            s.freq = self.pdf[i];
            s.cmpl_freq = (1 << PROB_BITS) - self.pdf[i];
            if self.pdf[i] < 2 {
                s.rcp_freq = !0_u64;
                s.rcp_shift = 0;
                s.bias = self.cdf[i] + (1 << PROB_BITS) - 1;
            } else {
                let mut shift = 0;
                while self.pdf[i] > (1 << shift) {
                    shift += 1;
                }
                let mut x0: u64 = self.pdf[i] as u64 - 1;
                let x1 = 1_u64 << (shift + 31);
                let t1 = x1 / self.pdf[i] as u64;
                x0 += (x1 % self.pdf[i] as u64) << 32;
                let t0 = x0 / self.pdf[i] as u64;

                s.rcp_freq = t0 + (t1 << 32);
                s.rcp_shift = shift - 1;

                s.bias = self.cdf[i];
            }
        }
    }

    /// Updates the probability distribution for the symbols. This can be done
    /// dynamically as the model of the data changes.
    pub fn update_probs(&mut self, probs: &[u32]) {
        // Probabilities are represented as u32, they should sum to target_total
        let target_total = 1 << PROB_BITS;
        self.probs.clone_from_slice(probs);

        // Compute the cumulative sum
        self.cdf[0] = 0;
        self.cdf[1..].clone_from_slice(&self.probs);
        self.cdf.iter_mut().fold(0, |acc, x| {
            *x += acc;
            *x
        });

        // Let normalize so we sum to target_total
        let total = self.cdf[self.cdf.len() - 1];
        for x in self.cdf.iter_mut().skip(1) {
            // We cast to u64 because the multiplication may overflow
            *x = ((*x as u64) * (target_total as u64) / total as u64) as u32;
        }

        for (i, &prob) in probs.iter().enumerate() {
            // If some probability was set to 0 by the division above, steal
            // from other low probability tokens to make it non-0.
            if prob > 0 && (self.cdf[i + 1] == self.cdf[i]) {
                let mut best_freq = !0;
                let mut best_steal = -1;

                // Find the lowest freq > 1 (aka the best to steal from)
                for j in 0..256 {
                    let freq = self.cdf[j + 1] - self.cdf[j];
                    if freq > 1 && freq < best_freq {
                        best_freq = freq;
                        best_steal = j as i16;
                    }
                }

                assert_ne!(best_steal, -1);

                // We adjust the rest of the cumulative dist to account for the
                // change
                if (best_steal as usize) < i {
                    for j in (best_steal + 1) as usize..=i {
                        self.cdf[j] -= 1;
                    }
                } else {
                    assert!(best_steal > i as i16);
                    for j in i + 1..=best_steal as usize {
                        self.cdf[j] += 1;
                    }
                }
            }
        }
        assert_eq!(self.cdf[0], 0);
        assert_eq!(self.cdf[256], target_total);

        for (i, &prob) in probs.iter().enumerate() {
            // Checks that everything went ok
            if prob == 0 {
                assert_eq!(self.cdf[i + 1], self.cdf[i]);
            } else {
                assert!(self.cdf[i + 1] > self.cdf[i]);
            }
            // Calc updated freq
            self.pdf[i] = self.cdf[i + 1] - self.cdf[i];
        }

        if let Mode::Precomp = self.model_mode {
            self.precomp()
        }
    }
}

pub struct ANSCoder {
    state: RansState,
    pub encoded_data: Vec<u32>,
    pub stats: SymbolStats,
}

impl ANSCoder {
    pub fn new() -> ANSCoder {
        ANSCoder {
            state: RANS64_L as RansState,
            encoded_data: vec![],
            stats: SymbolStats::new(),
        }
    }

    pub fn new_static(probs: &[u32]) -> ANSCoder {
        ANSCoder {
            state: RANS64_L as RansState,
            encoded_data: vec![],
            stats: SymbolStats::new_static(probs),
        }
    }

    pub fn new_precomp(probs: &[u32]) -> ANSCoder {
        ANSCoder {
            state: RANS64_L as RansState,
            encoded_data: vec![],
            stats: SymbolStats::new_precomp(probs),
        }
    }

    /// Encode a single symbol
    pub fn encode_symbol(&mut self, symbol: u8) {
        if let Mode::Precomp = self.stats.model_mode {
            self.encode_symbol_precomp(symbol);
        } else {
            let freq = self.stats.pdf[symbol as usize];
            let start = self.stats.cdf[symbol as usize];

            let mut x: RansState = self.state;
            let x_max = ((RANS64_L >> PROB_BITS) << 32) * freq as u64;

            if x >= x_max {
                self.encoded_data.push(x as u32);
                x >>= 32;
            }

            self.state = ((x / freq as u64) << PROB_BITS) + (x % freq as u64) + start as u64;
        }
    }

    /// Encode a single symbol in precomputed mode. Can be called directly if
    /// already in precomputed mode to avoid a comparison.
    pub fn encode_symbol_precomp(&mut self, symbol: u8) {
        let pre = self.stats.precomp[symbol as usize];
        let mut x: u64 = self.state;
        let x_max = ((RANS64_L >> PROB_BITS) << 32) * pre.freq as u64;
        if x >= x_max {
            self.encoded_data.push(x as u32);
            x >>= 32;
        }

        // Cast to u128 to avoid multiplication overflow
        let q = ((((x as u128) * (pre.rcp_freq as u128)) >> 64) >> pre.rcp_shift) as u64;
        self.state = x + pre.bias as u64 + q * pre.cmpl_freq as u64;
    }

    /// Obtain the encoded data so far
    pub fn get_encoded(&mut self) -> Vec<u32> {
        self.encoded_data.push(self.state as u32);
        self.state >>= 32;
        self.encoded_data.push(self.state as u32);
        self.encoded_data.clone()
    }
}

pub struct ANSDecoder {
    state: u64,
    pub encoded_data: Vec<u32>,
    pub stats: SymbolStats,
}

impl ANSDecoder {
    pub fn new(mut encoded_data: Vec<u32>) -> ANSDecoder {
        ANSDecoder {
            state: ((encoded_data.pop().unwrap() as u64) << 32)
                | encoded_data.pop().unwrap() as u64,
            encoded_data,
            stats: SymbolStats::new(),
        }
    }

    /// Decode a single symbol
    pub fn decode_symbol(&mut self) -> Result<u8, String> {
        let (pdf, cdf) = (&self.stats.pdf, &self.stats.cdf);
        let to_find = self.state & ((1 << PROB_BITS) - 1);
        let symbol = find_in_int_dist(cdf, to_find);

        let start = cdf[symbol as usize];
        let freq = pdf[symbol as usize];

        let mask = (1 << PROB_BITS) - 1;
        let mut x = self.state;
        x = freq as u64 * (x >> PROB_BITS) + (x & mask) - start as u64;

        if x < RANS64_L {
            x = (x << 32) | self.encoded_data.pop().unwrap() as u64
        }
        self.state = x;

        Ok(symbol as u8)
    }
}
