use std::fs::File;
use std::io::Read;
use std::time::Instant;

use rans::ANSCoder;

#[test]
fn decode_book1() {
    let mut data: Vec<u8> = vec![];

    // Read test file
    let mut f = File::open("book1").unwrap();
    f.read_to_end(&mut data).unwrap();

    // Compute probablities for every token in the document
    let mut probs = vec![0; 256];
    for c in data.iter() {
        probs[*c as usize] += 1;
    }
    let mut ans = ANSCoder::new_static(&probs);

    println!("Normal:");
    for _ in 0..5 {
        ans = ANSCoder::new_static(&probs);
        let now = Instant::now();
        for symbol in data.iter() {
            ans.encode_symbol(*symbol);
        }
        let dur = now.elapsed();
        println!(
            "\t{:.3} seconds elapsed, {:.3}MiB/sec",
            dur.as_millis() as f64 / 1000.,
            data.len() as f64 / (2_f64.powf(20.) * dur.as_nanos() as f64 / 1e9)
        );
    }

    println!("Update probs every 50 tokens:");
    for _ in 0..5 {
        ans = ANSCoder::new();
        let now = Instant::now();
        for (i, symbol) in data.iter().enumerate() {
            if i % 50 == 0 {
                ans.stats.update_probs(&probs);
            }
            ans.encode_symbol(*symbol);
        }
        let dur = now.elapsed();
        println!(
            "\t{:.3} seconds elapsed, {:.3}MiB/sec",
            dur.as_millis() as f64 / 1000.,
            data.len() as f64 / (2_f64.powf(20.) * dur.as_nanos() as f64 / 1e9)
        );
    }

    println!("Optimized:");
    for _ in 0..5 {
        ans = ANSCoder::new_precomp(&probs);
        let now = Instant::now();
        for symbol in data.iter() {
            ans.encode_symbol_precomp(*symbol);
        }
        let dur = now.elapsed();
        println!(
            "\t{:.3} seconds elapsed, {:.3}MiB/sec",
            dur.as_millis() as f64 / 1000.,
            data.len() as f64 / (2_f64.powf(20.) * dur.as_nanos() as f64 / 1e9)
        );
    }

    let encoded = ans.get_encoded();
    println!("Encoded data size {}", encoded.len() * 4);

    // Decode data
    let mut decoder = ANSDecoder::new(encoded);
    decoder.stats = ans.stats;
    let mut decoded_data = vec![];
    let length_decoded = data.len();

    for _ in 0..length_decoded {
        decoded_data.push(decoder.decode_symbol().unwrap())
    }
    decoded_data = decoded_data.into_iter().rev().collect();

    assert_eq!(data.len(), decoded_data.len());
    assert!(vec_compare(&data, &decoded_data));
    println!("Decoding ok!");
}
