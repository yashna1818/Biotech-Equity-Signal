import re
import numpy as np

class BiotechSentimentAnalyzer:
    """
    A sentiment analyzer tailored for biotechnology and financial text.
    It attempts to load FinBERT from HuggingFace, but falls back to a highly customized
    biotech-financial lexicon model if transformers are not available or fail to download.
    """
    def __init__(self, use_transformers: bool = True):
        self.use_transformers = use_transformers
        self.pipeline = None
        self.model_loaded = False
        
        if self.use_transformers:
            try:
                # Try to import and load ProsusAI/finbert
                from transformers import pipeline
                print("[Info] Attempting to load FinBERT model (ProsusAI/finbert)...")
                # Using the pipeline for sentiment classification
                self.pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                self.model_loaded = True
                print("[Success] FinBERT model loaded successfully.")
            except Exception as e:
                print(f"[Warning] Could not load FinBERT via Hugging Face. Fallback local lexicon analyzer will be used. Error: {e}")
                
        # Custom lexicon for Biotech and Financial sentiment mapping
        self.pos_words = {
            "approve", "approval", "approves", "approved", "efficacy", "effective", "safe", 
            "positive", "success", "successful", "successfully", "meet", "met", "breakthrough", 
            "gain", "rise", "profit", "growth", "earnings", "beat", "outperform", "expand", 
            "agreement", "partner", "partnership", "collaboration", "authorized", "authorization",
            "promising", "significant", "pipeline", "robust", "patent", "granted", "fda_approval",
            "trial_success", "phase_3_success", "clearance", "cleared"
        }
        
        self.neg_words = {
            "fail", "failure", "failed", "fails", "reject", "rejection", "rejected", "rejects", 
            "recall", "recalled", "warning", "warns", "warned", "adverse", "side-effect", 
            "toxicity", "toxic", "death", "hospitalization", "miss", "missed", "drop", 
            "fall", "loss", "decline", "deficit", "crl", "complete_response_letter", "investigation",
            "lawsuit", "disappointing", "concern", "concerns", "delay", "delayed", "trial_failure",
            "safety_warning", "drug_recall", "fda_rejection"
        }

    def analyze_sentiment(self, text: str):
        """
        Analyzes the sentiment of the text.
        Returns a dictionary: { "label": "positive"|"negative"|"neutral", "score": float }
        """
        if not text:
            return {"label": "neutral", "score": 1.0}
            
        if self.model_loaded and self.pipeline:
            try:
                # FinBERT outputs: label: 'positive'/'negative'/'neutral', score: probability
                result = self.pipeline(text)[0]
                return {
                    "label": result["label"].lower(),
                    "score": float(result["score"])
                }
            except Exception as e:
                # Fallback on inference error
                pass
                
        # Rules-based Fallback Sentiment Analyzer
        # Normalize text and split to words
        words = re.findall(r'\b\w+\b', text.lower())
        
        pos_count = sum(1 for w in words if w in self.pos_words)
        neg_count = sum(1 for w in words if w in self.neg_words)
        
        # Calculate scores
        total = pos_count + neg_count
        if total == 0:
            return {"label": "neutral", "score": 0.5}
            
        score_diff = pos_count - neg_count
        confidence = abs(score_diff) / total
        
        # Smooth confidence score between 0.5 and 1.0
        confidence = 0.5 + (confidence * 0.5)
        
        if score_diff > 0:
            return {"label": "positive", "score": confidence}
        elif score_diff < 0:
            return {"label": "negative", "score": confidence}
        else:
            return {"label": "neutral", "score": 0.5}

    def detect_events(self, text: str):
        """
        Detects regulatory and clinical events in the text.
        Returns a dictionary of event flags, event type, and severity.
        """
        text_lower = text.lower()
        
        # Flags init
        events = {
            "fda_approval": 0,
            "trial_success": 0,
            "trial_failure": 0,
            "drug_recall": 0,
            "safety_warning": 0,
            "has_event": 0,
            "event_type": "None",
            "event_severity": 0.0 # scale of 0 to 1
        }
        
        # 1. FDA Approval Detection
        if any(term in text_lower for term in ["fda_approval", "fda approves", "fda approved", "food and drug administration approved", "marketing approval", "receives approval"]):
            events["fda_approval"] = 1
            events["event_type"] = "FDA Approval"
            events["event_severity"] = 0.8
            
        # 2. Trial Success Detection
        if any(term in text_lower for term in ["trial_success", "trial successful", "met primary endpoint", "efficacy endpoints met", "positive phase 3 results"]):
            events["trial_success"] = 1
            # If FDA approval is already set, don't overwrite event_type unless priority
            if events["event_type"] == "None":
                events["event_type"] = "Clinical Trial Success"
                events["event_severity"] = 0.7
                
        # 3. Trial Failure Detection
        if any(term in text_lower for term in ["trial_failure", "trial failed", "failed to meet", "missed primary endpoint", "efficacy endpoint missed"]):
            events["trial_failure"] = 1
            events["event_type"] = "Clinical Trial Failure"
            events["event_severity"] = -0.7
            
        # 4. Drug Recall Detection
        if any(term in text_lower for term in ["drug_recall", "drug recalled", "fda recall", "market withdrawal", "recalled from"]):
            events["drug_recall"] = 1
            events["event_type"] = "Drug Recall"
            events["event_severity"] = -0.9
            
        # 5. Safety Warning Detection
        if any(term in text_lower for term in ["safety_warning", "safety warning", "adverse event", "black box warning", "side effect", "cardiovascular risk"]):
            events["safety_warning"] = 1
            if events["event_type"] == "None" or events["event_severity"] > -0.5:
                events["event_type"] = "Safety Warning"
                events["event_severity"] = -0.5
                
        # Flag if any event is detected
        if any([events["fda_approval"], events["trial_success"], events["trial_failure"], events["drug_recall"], events["safety_warning"]]):
            events["has_event"] = 1
            
        return events

if __name__ == "__main__":
    # Test sentiment and event engine
    analyzer = BiotechSentimentAnalyzer(use_transformers=False)
    
    test_cases = [
        "FDA approves groundbreaking cancer therapy for BiotechCorp.",
        "Pfizer shares tumble after clinical trial fails to meet primary efficacy endpoint.",
        "FDA issues black box warning on blood pressure drug due to side effects.",
        "Moderna reports steady earnings in line with market consensus.",
    ]
    
    for tc in test_cases:
        sentiment = analyzer.analyze_sentiment(tc)
        events = analyzer.detect_events(tc)
        print(f"\nText: {tc}")
        print(f"Sentiment: {sentiment}")
        print(f"Event Detected: {events['event_type']} (Severity: {events['event_severity']})")
