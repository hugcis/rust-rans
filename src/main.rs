fn main() -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::*;
    use rand::Rng;
    use std::fs::File;
    use std::io::Read;
    use std::time::Instant;

    use rans::{ANSCoder, ANSDecoder};

    fn vec_compare<N: std::cmp::PartialEq + Copy>(va: &[N], vb: &[N]) -> bool {
        va.iter().zip(vb).all(|(&a, &b)| a == b)
    }

    #[test]
    fn encode_decode_random() {
        let mut rng = rand::thread_rng();
        let mut ans = ANSCoder::new();
        let probs = vec![1; 256];
        ans.stats.update_probs(&probs);
        let mut data: Vec<u8> = vec![];

        for _ in 0..10000 {
            data.push(rng.gen_range(0..8));
        }

        for symbol in data.iter() {
            ans.encode_symbol(*symbol)
        }

        let encoded = ans.get_encoded();
        let mut decoder = ANSDecoder::new(encoded);
        decoder.stats = ans.stats;
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
