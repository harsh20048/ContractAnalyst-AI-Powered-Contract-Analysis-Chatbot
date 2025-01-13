# document_analyzer.py

import re
import nltk
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK and spaCy data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section in the document"""
    title: str
    content: str
    level: int
    parent: Optional[str]
    children: List[str]
    start_page: int
    definitions: Dict[str, str]

@dataclass
class DocumentStructure:
    """Represents the complete document structure"""
    sections: Dict[str, DocumentSection]
    definitions: Dict[str, str]
    hierarchy: Dict[str, List[str]]
    section_refs: Dict[str, List[str]]

class DocumentAnalyzer:
    def __init__(self):
        self.section_pattern = re.compile(r'^(?P<number>[\d.]+)\s+(?P<title>[A-Z][^.]+)')
        self.definition_pattern = re.compile(r'"(?P<term>[^"]+)"\s+means\s+(?P<definition>[^.]+)')
        self.reference_pattern = re.compile(r'Section\s+[\d.]+')
        
    def extract_structure(self, text: str) -> DocumentStructure:
        """Extract document structure including sections, definitions, and references"""
        sections = {}
        definitions = {}
        hierarchy = defaultdict(list)
        current_section = None
        current_level = 0
        
        # Split text into lines and process
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Match section headers
            section_match = self.section_pattern.match(line)
            if section_match:
                section_num = section_match.group('number')
                section_title = section_match.group('title')
                
                # Determine section level and parent
                level = len(section_num.split('.'))
                parent = '.'.join(section_num.split('.')[:-1]) if level > 1 else None
                
                # Create section object
                sections[section_num] = DocumentSection(
                    title=section_title,
                    content='',
                    level=level,
                    parent=parent,
                    children=[],
                    start_page=self._estimate_page_number(i, len(lines)),
                    definitions={}
                )
                
                # Update hierarchy
                if parent:
                    hierarchy[parent].append(section_num)
                    if parent in sections:
                        sections[parent].children.append(section_num)
                
                current_section = section_num
                current_level = level
                
            # Process section content
            elif current_section and line.strip():
                sections[current_section].content += line + '\n'
                
                # Extract definitions
                def_match = self.definition_pattern.search(line)
                if def_match:
                    term = def_match.group('term')
                    definition = def_match.group('definition')
                    definitions[term] = definition
                    sections[current_section].definitions[term] = definition
        
        # Extract cross-references
        section_refs = self._extract_cross_references(text)
        
        return DocumentStructure(sections, definitions, dict(hierarchy), section_refs)
    
    def _estimate_page_number(self, line_num: int, total_lines: int) -> int:
        """Estimate page number based on line position"""
        LINES_PER_PAGE = 50  # Approximate
        return (line_num // LINES_PER_PAGE) + 1
    
    def _extract_cross_references(self, text: str) -> Dict[str, List[str]]:
        """Extract cross-references between sections"""
        references = defaultdict(list)
        for match in self.reference_pattern.finditer(text):
            ref = match.group()
            section = self._get_current_section(text[:match.start()])
            if section:
                references[section].append(ref)
        return dict(references)
    
    def _get_current_section(self, text: str) -> Optional[str]:
        """Determine the current section based on text position"""
        sections = self.section_pattern.finditer(text)
        current = None
        for match in sections:
            current = match.group('number')
        return current
    
    def search_semantic(self, query: str, doc_structure: DocumentStructure) -> List[Tuple[str, float]]:
        """Perform semantic search across document sections"""
        query_doc = nlp(query)
        results = []
        
        for section_id, section in doc_structure.sections.items():
            section_doc = nlp(section.content)
            similarity = query_doc.similarity(section_doc)
            results.append((section_id, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def generate_summary(self, doc_structure: DocumentStructure) -> str:
        """Generate a comprehensive document summary"""
        summary = []
        
        # Add document overview
        summary.append("Document Overview:")
        top_level_sections = [s for s in doc_structure.sections.values() if s.level == 1]
        for section in top_level_sections:
            summary.append(f"- {section.title}")
        
        # Add key definitions
        summary.append("\nKey Definitions:")
        for term, definition in doc_structure.definitions.items():
            summary.append(f"- {term}: {definition}")
        
        # Add section highlights
        summary.append("\nKey Sections:")
        for section_id, section in doc_structure.sections.items():
            if section.level <= 2:  # Only include top 2 levels
                summary.append(f"{section_id}. {section.title}")
        
        return "\n".join(summary)
    
    def handle_followup(self, query: str, context: str, doc_structure: DocumentStructure) -> str:
        """Handle follow-up questions using context"""
        # Extract section references from context
        context_sections = self.reference_pattern.findall(context)
        
        # Look for relative references in query
        relative_refs = {"this section", "that section", "it", "this"}
        has_relative_ref = any(ref in query.lower() for ref in relative_refs)
        
        if has_relative_ref and context_sections:
            # Use the most recently mentioned section
            current_section = context_sections[-1]
            # Update query to include explicit section reference
            query = query.replace("this section", current_section)
            query = query.replace("that section", current_section)
        
        return self.search_semantic(query, doc_structure)

class ResponseGenerator:
    """Generates user-friendly responses for document queries"""
    
    def __init__(self, doc_structure: DocumentStructure):
        self.doc_structure = doc_structure
    
    def get_section_content(self, section_id: str) -> str:
        """Get formatted content for a section"""
        section = self.doc_structure.sections.get(section_id)
        if not section:
            return "Section not found."
        
        response = [f"Section {section_id}: {section.title}\n"]
        response.append(section.content)
        
        if section.definitions:
            response.append("\nDefinitions in this section:")
            for term, definition in section.definitions.items():
                response.append(f"- {term}: {definition}")
        
        return "\n".join(response)
    
    def get_definition(self, term: str) -> str:
        """Get definition for a term"""
        if term in self.doc_structure.definitions:
            return f'"{term}" means {self.doc_structure.definitions[term]}'
        return f"Definition for '{term}' not found in the document."
    
    def generate_outline(self) -> str:
        """Generate document outline"""
        outline = ["Document Outline:"]
        
        def add_section(section_id: str, level: int = 0):
            section = self.doc_structure.sections[section_id]
            indent = "  " * level
            outline.append(f"{indent}- {section_id}. {section.title}")
            for child in section.children:
                add_section(child, level + 1)
        
        # Add top-level sections
        for section_id, section in self.doc_structure.sections.items():
            if section.level == 1:
                add_section(section_id)
        
        return "\n".join(outline)
    
    def handle_error(self, query: str) -> str:
        """Generate helpful error response"""
        # Find most relevant sections
        query_terms = set(word_tokenize(query.lower()))
        relevant_sections = []
        
        for section_id, section in self.doc_structure.sections.items():
            section_terms = set(word_tokenize(section.title.lower()))
            if query_terms & section_terms:
                relevant_sections.append((section_id, section.title))
        
        response = ["I couldn't find an exact match for your query."]
        
        if relevant_sections:
            response.append("\nYou might find relevant information in these sections:")
            for section_id, title in relevant_sections[:3]:
                response.append(f"- Section {section_id}: {title}")
        
        return "\n".join(response)