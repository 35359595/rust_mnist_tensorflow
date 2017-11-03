//extern crate flate2;
use std::io::Read;
use std::fs::File;
use flate2::read::GzDecoder;

pub fn import_data(path: &'static str) -> Vec<u8> {
    let mut content = Vec::new();
    let mut file = File::open(path)
        .expect(&format!("Failed to open file: {}", path)[..]);

    GzDecoder::new(file).unwrap().read_to_end(&mut content);
    return content
}
