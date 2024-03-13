// For hybrid compute to work, [no_mangle] must be at the top of all functions
// we plan to call from our quantum program
// For more info see: [Calling Rust from Other Languages]
// (https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html?highlight=no_mangle#calling-rust-functions-from-other-languages)
#[no_mangle]
fn init(){
    // This function can have nothing it in, or load some initial function. 
    // It is needed when passed via the Quantinuum API to warm up the wasm execution environment. 
    // It can also be used to set up a global state.
}

fn get_bit(bit_string: u8, index: u32) -> u8 {
    let base: u8 = 2;
    (bit_string & base.pow(index)) >> index
}

fn get_correction(initial_pauli: u8) -> u8 {
    
    // calculate x_1
    let mut correct_pauli: u8 = get_bit(initial_pauli, 3);
    correct_pauli = correct_pauli << 1;

    // calculate x_0
    correct_pauli = correct_pauli ^ get_bit(initial_pauli, 2);
    correct_pauli = correct_pauli << 1;

    // calculate z_1
    correct_pauli = correct_pauli ^ get_bit(initial_pauli, 1);
    correct_pauli = correct_pauli ^ get_bit(initial_pauli, 2);
    correct_pauli = correct_pauli ^ get_bit(initial_pauli, 3);
    correct_pauli = correct_pauli << 1;

    // calculate z_0
    correct_pauli = correct_pauli ^ get_bit(initial_pauli, 0);
    correct_pauli = correct_pauli ^ get_bit(initial_pauli, 2);
    correct_pauli = correct_pauli ^ get_bit(initial_pauli, 3);

    correct_pauli
}

#[no_mangle]
pub extern "C" fn write_randomisation() -> u8 {
    
    let initial_pauli: u8 = fastrand::u8(..) >> 4;
    let correct_pauli: u8 = get_correction(initial_pauli);

    initial_pauli | correct_pauli << 4
}

#[no_mangle]
pub extern "C" fn seed_randomisation(seed: u16) {
    fastrand::seed(seed as u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_correction() {
        assert_eq!(get_correction(14), 14);
        assert_eq!(get_correction(9), 10);
        assert_eq!(get_correction(4), 7);
        assert_eq!(get_correction(2), 2);
        assert_eq!(get_correction(6), 5);
    }
}
