import re
from datetime import datetime
import pandas as pd
import unittest
from text_analysis import TextAnalyzer

class TextAnalyzer:
    def __init__(self):
        self.text = ""
        self.analysis_results = {}
        
    def extract_entities(self, text):
        doc = self.nlp(text)
        
        # Extract dates
        date_pattern = r'\d{2}/\d{2}/\d{4}'
        dates = re.findall(date_pattern, text)
        
        # Extract names of parties
        parties = [ent.text for ent in doc.ents if ent.label_ == 'PER']
        
        # Extract decision segments
        decision_keywords = ['DECIDO', 'DECISÃO', 'SENTENÇA']
        decisions = []
        for sentence in doc.sents:
            if any(keyword in sentence.text.upper() for keyword in decision_keywords):
                decisions.append(sentence.text.strip())
                
        return {
            'dates': dates,
            'parties': parties,
            'decisions': decisions
        }
    
    def classify_complexity(self, text):
        # Simple complexity classification based on text features
        features = {
            'length': len(text),
            'num_parties': len(self._extract_parties(text)),
            'num_laws': len(self._extract_laws(text)),
            'num_dates': len(self._extract_dates(text))
        }
        
        # Basic scoring system
        score = (
            features['length'] / 1000 +
            features['num_parties'] * 2 +
            features['num_laws'] * 3 +
            features['num_dates']
        )
        
        if score < 10:
            return "Baixa"
        elif score < 20:
            return "Média"
        else:
            return "Alta"
    
    def _extract_parties(self, text):
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == 'PER']
    
    def _extract_laws(self, text):
        law_pattern = r'Lei\s+[nN]º\s+[\d\.]+'
        return re.findall(law_pattern, text)
    
    def _extract_dates(self, text):
        date_pattern = r'\d{2}/\d{2}/\d{4}'
        return re.findall(date_pattern, text)
    
    def analyze_text(self, text):
        """Analyze the given text"""
        self.text = text
        # Add analysis logic here
        return {
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def get_summary(self):
        """Return summary of analysis"""
        if not self.text:
            return "No text analyzed yet"
        return f"Text length: {len(self.text)} characters"

    def analyze(self, text):
        """Analyze input text and return basic metrics"""
        self.text = text
        self.analysis_results = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'line_count': len(text.splitlines()),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text else 0
        }
        return self.analysis_results

    def get_results(self):
        """Return current analysis results"""
        return self.analysis_results

class TestTextAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()
        
    def test_extract_entities(self):
        text = """
        No dia 01/02/2024, João Silva e Maria Santos compareceram à audiência.
        DECISÃO: Processo deferido.
        """
        result = self.analyzer.extract_entities(text)
        
        self.assertIn('01/02/2024', result['dates'])
        self.assertTrue(any('João Silva' in party for party in result['parties']))
        self.assertTrue(any('Processo deferido' in decision for decision in result['decisions']))