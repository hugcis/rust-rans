use std::fs::File;
use std::io::Read;
use std::time::Instant;

const PROB_BITS: u32 = 14;
const RANS64_L: u64 = 1 << 31;

type RansState = u64;

fn vec_compare_f64(va: &[f64], vb: &[f64]) -> bool {
    va.iter().zip(vb).all(|(a, b)| (a - b).abs() < 1E-12)
}

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
    STATIC,
    PRECOMP,
    DYN,
}

#[derive(Default, Clone, Copy)]
struct PreCompSym {
    rcp_freq: u64,
    freq: u32,
    bias: u32,
    cmpl_freq: u32,
    rcp_shift: u32,
}

struct SymbolStats {
    pdf: [u32; 256],
    cdf: [u32; 257],
    probs: [u32; 256],
    model_mode: Mode,
    precomp: [PreCompSym; 256],
}

impl SymbolStats {
    fn new() -> SymbolStats {
        SymbolStats {
            pdf: [0; 256],
            cdf: [0; 257],
            probs: [0; 256],
            model_mode: Mode::DYN,
            precomp: [PreCompSym::default(); 256],
        }
    }

    fn new_static(probs: &[u32]) -> SymbolStats {
        let mut stats = SymbolStats {
            pdf: [0; 256],
            cdf: [0; 257],
            probs: [0; 256],
            model_mode: Mode::STATIC,
            precomp: [PreCompSym::default(); 256],
        };
        stats.update_probs(probs);
        stats
    }

    fn new_precomp(probs: &[u32]) -> SymbolStats {
        let mut stats = SymbolStats {
            pdf: [0; 256],
            cdf: [0; 257],
            probs: [0; 256],
            model_mode: Mode::STATIC,
            precomp: [PreCompSym::default(); 256],
        };
        stats.update_probs(probs);
        stats
    }

    fn precomp(&mut self) {
        for i in 0..256 {
            self.precomp[i].freq = self.pdf[i];
            self.precomp[i].cmpl_freq = (1 << PROB_BITS) - self.pdf[i];
        }
    }

    fn update_probs(&mut self, probs: &[u32]) {
        let target_total = 1 << PROB_BITS;
        self.probs.clone_from_slice(probs);
        self.cdf[1..].clone_from_slice(&self.probs);
        self.cdf.iter_mut().fold(0, |acc, x| {
            *x += acc;
            *x
        });
        let total = self.cdf[self.cdf.len() - 1];
        for x in self.cdf.iter_mut().skip(1) {
            *x = ((*x as u64) * (target_total as u64) / total as u64) as u32;
        }
        for i in 0..256 {
            if probs[i] > 0 && (self.cdf[i + 1] == self.cdf[i]) {
                let mut best_freq = !0;
                let mut best_steal = -1;

                for j in 0..256 {
                    let freq = self.cdf[j + 1] - self.cdf[j];
                    if freq > 1 && freq < best_freq {
                        best_freq = freq;
                        best_steal = j as i16;
                    }
                }

                assert_ne!(best_steal, -1);

                if (best_steal as usize) < i {
                    for j in (best_steal + 1) as usize..=i {
                        self.cdf[j] -= 1;
                    }
                } else {
                    assert_eq!(best_steal > i as i16, true);
                    for j in i + 1..=best_steal as usize {
                        self.cdf[j] += 1;
                    }
                }
            }
        }
        assert_eq!(self.cdf[0], 0);
        assert_eq!(self.cdf[256], target_total);
        for i in 0..256 {
            if probs[i] == 0 {
                assert_eq!(self.cdf[i + 1], self.cdf[i]);
            } else {
                assert_eq!(self.cdf[i + 1] > self.cdf[i], true);
            }
            // calc updated freq
            self.pdf[i] = self.cdf[i + 1] - self.cdf[i];
        }
    }
}

struct ANSCoder {
    state: u64,
    encoded_data: Vec<u32>,
    stats: SymbolStats,
}

impl ANSCoder {
    fn new() -> ANSCoder {
        ANSCoder {
            state: RANS64_L as u64,
            encoded_data: vec![],
            stats: SymbolStats::new(),
        }
    }

    fn new_static(probs: &[u32]) -> ANSCoder {
        ANSCoder {
            state: RANS64_L as u64,
            encoded_data: vec![],
            stats: SymbolStats::new_static(probs),
        }
    }

    fn encode_symbol(&mut self, symbol: u8) {
        let freq = self.stats.pdf[symbol as usize];
        let start = self.stats.cdf[symbol as usize];

        let mut x: u64 = self.state;
        let x_max = ((RANS64_L >> PROB_BITS) << 32) * freq as u64;

        if x >= x_max {
            self.encoded_data.push(x as u32);
            x >>= 32;
        }

        self.state = ((x / freq as u64) << PROB_BITS) + (x % freq as u64) + start as u64;
    }

    fn get_encoded(&mut self) -> Vec<u32> {
        self.encoded_data.push(self.state as u32);
        self.state >>= 32;
        self.encoded_data.push(self.state as u32);
        self.encoded_data.clone()
    }
}

struct ANSDecoder {
    state: u64,
    encoded_data: Vec<u32>,
    float_to_int: SymbolStats,
}

impl ANSDecoder {
    fn new(mut encoded_data: Vec<u32>) -> ANSDecoder {
        ANSDecoder {
            state: ((encoded_data.pop().unwrap() as u64) << 32)
                | encoded_data.pop().unwrap() as u64,
            encoded_data,
            float_to_int: SymbolStats::new(),
        }
    }

    fn decode_symbol(&mut self) -> Result<u64, String> {
        let (pdf, cdf) = (&self.float_to_int.pdf, &self.float_to_int.cdf);
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

        Ok(symbol)
    }
}

fn main() -> std::io::Result<()> {
    println!("Hello, world!");
    let mut probs = vec![0; 256];

    let mut data: Vec<u8> = vec![];

    let mut f = File::open("book1")?;
    f.read_to_end(&mut data)?;

    for c in data.iter() {
        probs[*c as usize] += 1;
    }
    let mut ans = ANSCoder::new_static(&probs);

    for _ in 0..5 {
        ans = ANSCoder::new_static(&probs);
        ans.encoded_data = Vec::with_capacity(data.len());
        let now = Instant::now();
        for symbol in data.iter() {
            ans.encode_symbol(*symbol);
        }
        let dur = now.elapsed();
        println!(
            "{} seconds elapsed, {:.5}MB/sec",
            dur.as_millis() as f64 / 1000.,
            data.len() as f64 / (2_f64.powf(20.) * dur.as_nanos() as f64 / 1e9)
        );
    }

    let encoded = ans.get_encoded();
    println!("Encoded data size {}", encoded.len() * 4);

    let mut decoder = ANSDecoder::new(encoded);
    decoder.float_to_int = ans.stats;
    let mut decoded_data = vec![];
    let length_decoded = data.len();

    for _ in 0..length_decoded {
        decoded_data.push(decoder.decode_symbol().unwrap())
    }
    decoded_data = decoded_data.into_iter().rev().collect();

    assert_eq!(data.len(), decoded_data.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::*;

    fn vec_compare(va: &[u64], vb: &[u64]) -> bool {
        va.iter().zip(vb).all(|(a, b)| a == b)
    }

    #[test]
    fn encode_decode() {
        let mut rng = rand::thread_rng();
        let mut ans = ANSCoder::new();
        let probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125];
        let mut data: Vec<u8> = vec![];

        for _ in 0..10000 {
            data.push(rng.gen_range(0..8));
        }

        for symbol in data.iter() {
            ans.encode_symbol(*symbol)
        }

        let encoded = ans.get_encoded();
        let mut decoder = ANSDecoder::new(encoded);
        let mut decoded_data = vec![];
        let length_decoded = data.len();
        for _ in 0..length_decoded {
            decoded_data.push(decoder.decode_symbol().unwrap())
        }
        decoded_data = decoded_data.into_iter().rev().collect();

        assert_eq!(data.len(), decoded_data.len());
        assert_eq!(vec_compare(&data, &decoded_data), true);
    }
}
