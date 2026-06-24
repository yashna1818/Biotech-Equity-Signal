import re
import string

# Hardcoded list of common stopwords as a fallback in case nltk downloads fail
DEFAULT_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
    "for", "with", "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
    "will", "just", "don", "should", "now"
}

class TextPreprocessor:
    """
    A robust preprocessor for biomedical and financial text.
    Handles cleaning, tokenization, stopword removal, lemmatization (with fallback),
    and domain terminology normalization.
    """
    def __init__(self):
        # Initialize NLTK libraries if available, otherwise use default fallback mechanisms
        self.stopwords = DEFAULT_STOPWORDS
        self.nltk_available = False
        
        try:
            import nltk
            # Try downloading resources silently
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.nltk_available = True
        except Exception as e:
            print(f"[Warning] NLTK could not be fully loaded, using internal fallback preprocessor. Error: {e}")
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        Removes HTML tags, punctuation, and extra whitespaces.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags using simple regex
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize double quotes, smart quotes, etc.
        text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
        
        # Remove punctuation except hyphen and slash (which might be important in biotech, e.g., Phase-3 or A/B)
        # Keep letters, numbers, and basic punctuation
        allowed_chars = string.ascii_letters + string.digits + " -/%"
        text = "".join([c if c in allowed_chars else " " for c in text])
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def normalize_terminology(self, text: str) -> str:
        """
        Standardizes specific biomedical and regulatory terms to unified formats
        so that subsequent NLP or event detection is robust.
        """
        if not text:
            return ""
        
        # Lowercase for uniform replacement matching
        normalized = text.lower()
        
        # Normalize Clinical Trial Phases
        phase_patterns = [
            (r'\b(phase\s*(?:3|iii|three))\b', 'phase_3'),
            (r'\b(phase\s*(?:2|ii|two))\b', 'phase_2'),
            (r'\b(phase\s*(?:1|i|one))\b', 'phase_1'),
            (r'\b(phase\s*(?:4|iv|four))\b', 'phase_4'),
        ]
        for pattern, replacement in phase_patterns:
            normalized = re.sub(pattern, replacement, normalized)
            
        # Normalize Regulatory approvals/warnings
        regulatory_patterns = [
            (r'\b(fda approved|fda approval|approved by the fda)\b', 'fda_approval'),
            (r'\b(fda rejected|fda rejection|complete response letter|crl|not approved)\b', 'fda_rejection'),
            (r'\b(clinical trial success|met primary endpoint|successful trial|met endpoint)\b', 'trial_success'),
            (r'\b(clinical trial failure|failed to meet|missed primary endpoint|failed trial|did not meet endpoint)\b', 'trial_failure'),
            (r'\b(safety warning|adverse event|side effect|black box warning)\b', 'safety_warning'),
            (r'\b(drug recall|recalled by fda|market withdrawal)\b', 'drug_recall'),
        ]
        for pattern, replacement in regulatory_patterns:
            normalized = re.sub(pattern, replacement, normalized)
            
        return normalized

    def lemmatize_word(self, word: str) -> str:
        """
        Lemmatizes a word if NLTK is available, otherwise applies a basic suffix stemmer.
        """
        if self.nltk_available and self.lemmatizer:
            try:
                return self.lemmatizer.lemmatize(word)
            except Exception:
                pass
        
        # Simple rule-based suffix fallback to strip common plurals/gerunds
        if len(word) > 4:
            if word.endswith('s') and not word.endswith('ss'):
                return word[:-1]
            elif word.endswith('ing'):
                return word[:-3]
            elif word.endswith('ed'):
                return word[:-2]
        return word

    def preprocess(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        """
        Runs the full text preprocessing pipeline:
        1. Clean HTML and punctuation
        2. Normalize terminology
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize
        """
        cleaned = self.clean_text(text)
        normalized = self.normalize_terminology(cleaned)
        
        # Tokenization (split by spaces)
        tokens = normalized.split()
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Remove stopwords
            if remove_stopwords and token in self.stopwords:
                continue
            
            # Lemmatize
            if lemmatize:
                token = self.lemmatize_word(token)
                
            processed_tokens.append(token)
            
        return " ".join(processed_tokens)

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    sample_text = "The FDA approved BioTechCorp's new drug after a successful Phase III trial! However, side effects were noted."
    print("Original Text:", sample_text)
    print("Preprocessed :", preprocessor.preprocess(sample_text))
