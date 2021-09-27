use rand::Rng;
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const PROB_BITS: u64 = 14;
const MIN_PROB: u64 = 8;
const RANS64_L: u64 = 1 << 31;
const MAX_PROB: u64 = (RANS64_L >> PROB_BITS) << 32;

type RansState = u64;

fn vec_compare_f64(va: &[f64], vb: &[f64]) -> bool {
    va.iter().zip(vb).all(|(a, b)| (a - b).abs() < 1E-12)
}

fn find_in_int_dist(cdf: &[u64], to_find: u64) -> Result<u64, String> {
    (0..cdf.len() - 1)
        .into_iter()
        .find(|&i| (cdf[i] <= to_find) && (cdf[i + 1] > to_find))
        .map(|x| x as u64)
        .ok_or_else(|| "Error finding symbol in dist".to_string())
}

struct FloatToInt {
    pdf: Vec<u64>,
    cdf: Vec<u64>,
    probs: Vec<f64>,
    memoize: bool,
}

impl FloatToInt {
    fn new() -> FloatToInt {
        FloatToInt {
            pdf: vec![],
            cdf: vec![],
            probs: vec![],
            memoize: true,
        }
    }

    fn float_to_int_probs(&mut self, float_probs: &[f64]) -> (&[u64], &[u64]) {
        if self.memoize && (float_probs.len() != self.probs.len())
            || !vec_compare_f64(float_probs, &self.probs)
        {
            let (pdf, cdf) = self.new_float_to_int_probs(float_probs);
            self.pdf = pdf;
            self.cdf = cdf;
            self.probs = float_probs.to_vec();
        }
        (&self.pdf, &self.cdf)
    }

    fn new_float_to_int_probs(&mut self, float_probs: &[f64]) -> (Vec<u64>, Vec<u64>) {
        let mut pdf = vec![];
        let mut cdf = vec![0];

        for prob in float_probs.iter() {
            let mut next_prob: u64 = (prob * (1 << PROB_BITS) as f64) as u64;
            if prob > &0. && next_prob < MIN_PROB {
                next_prob = MIN_PROB;
            }
            pdf.push(next_prob);
            cdf.push(cdf.last().unwrap() + next_prob);
        }

        let to_correct = (1 << PROB_BITS) - cdf.last().unwrap();
        let largest_idx = pdf
            .iter()
            .enumerate()
            .max_by_key(|(_, &value)| value)
            .map(|(idx, _)| idx)
            .unwrap();
        pdf[largest_idx] += to_correct;

        for val in cdf.iter_mut().skip(largest_idx + 1) {
            *val += to_correct;
        }

        (pdf, cdf)
    }
}

struct ANSCoder {
    state: u64,
    encoded_data: Vec<u64>,
    float_to_int: FloatToInt,
}

impl ANSCoder {
    fn new() -> ANSCoder {
        ANSCoder {
            state: RANS64_L,
            encoded_data: vec![],
            float_to_int: FloatToInt::new(),
        }
    }

    fn encode_symbol(&mut self, freqs: &[f64], symbol: u8) {
        let (pdf, cdf) = self.float_to_int.float_to_int_probs(&freqs);
        let freq = pdf[symbol as usize];
        let start = cdf[symbol as usize];

        let mut x = self.state;
        let x_max = MAX_PROB * freq;

        if x >= x_max {
            self.encoded_data.push(x & 0xffffffff);
            x >>= 32;
        }

        self.state = ((x / freq) << PROB_BITS) + (x % freq) + start;
    }

    fn get_encoded(&mut self) -> Vec<u64> {
        self.encoded_data.push(self.state & 0xffffffff);
        self.state >>= 32;
        self.encoded_data.push(self.state & 0xffffffff);
        self.encoded_data.clone()
    }
}

struct ANSDecoder {
    state: u64,
    encoded_data: Vec<u64>,
    float_to_int: FloatToInt,
}

impl ANSDecoder {
    fn new(mut encoded_data: Vec<u64>) -> ANSDecoder {
        ANSDecoder {
            state: (encoded_data.pop().unwrap() << 32) | encoded_data.pop().unwrap(),
            encoded_data,
            float_to_int: FloatToInt::new(),
        }
    }

    fn decode_symbol(&mut self, freqs: Vec<f64>) -> Result<u64, String> {
        let (pdf, cdf) = self.float_to_int.float_to_int_probs(&freqs);
        let to_find = self.state & ((1 << PROB_BITS) - 1);
        let symbol = find_in_int_dist(cdf, to_find)?;

        let start = cdf[symbol as usize];
        let freq = pdf[symbol as usize];

        let mask = (1 << PROB_BITS) - 1;
        let mut x = self.state;
        x = freq * (x >> PROB_BITS) + (x & mask) - start;

        if x < RANS64_L {
            x = (x << 32) | self.encoded_data.pop().unwrap()
        }
        self.state = x;

        Ok(symbol)
    }
}

fn main() -> std::io::Result<()> {
    println!("Hello, world!");
    let mut ans = ANSCoder::new();
    let mut rng = rand::thread_rng();
    let mut probs = vec![0.; 256];

    let mut data: Vec<u8> = vec![];
    let mut f = File::open("book1")?;
    f.read_to_end(&mut data)?;
    for c in data.iter() {
        probs[*c as usize] += 1.;
    }
    let sum = probs.iter().sum::<f64>();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    for _ in 0..5 {
        ans = ANSCoder::new();
        let now = Instant::now();
        for symbol in data.iter() {
            ans.encode_symbol(&probs, *symbol);
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
    let mut decoded_data = vec![];
    let length_decoded = data.len();
    for _ in 0..length_decoded {
        decoded_data.push(decoder.decode_symbol(probs.to_vec()).unwrap())
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
        let mut data: Vec<u64> = vec![];

        for _ in 0..10000 {
            data.push(rng.gen_range(0..8));
        }

        for symbol in data.iter() {
            ans.encode_symbol(&probs, (*symbol).try_into().unwrap())
                .unwrap();
        }

        let encoded = ans.get_encoded();
        let mut decoder = ANSDecoder::new(encoded);
        let mut decoded_data = vec![];
        let length_decoded = data.len();
        for _ in 0..length_decoded {
            decoded_data.push(decoder.decode_symbol(probs.to_vec()).unwrap())
        }
        decoded_data = decoded_data.into_iter().rev().collect();

        assert_eq!(data.len(), decoded_data.len());
        assert_eq!(vec_compare(&data, &decoded_data), true);
    }
}
