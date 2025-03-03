def convert(sign):
    """Convert recognized sign to text (if needed for further processing)."""
    # Example: Modify mapping if needed
    sign_text_mapping = {
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", 
        "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
        "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", 
        "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
        "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z",
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"
    }
    
    return sign_text_mapping.get(sign, "Unknown")
