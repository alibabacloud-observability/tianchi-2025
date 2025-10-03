#!/usr/bin/env python
"""
Evidence Chain implementation for agents
Compatible with the existing agent interfaces
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Evidence:
    """Individual evidence item"""
    evidence_type: str
    source: str
    data: Any
    timestamp: datetime = None
    confidence: float = 1.0


class EvidenceChain:
    """Evidence Chain for collecting and organizing analysis evidence"""
    
    def __init__(self, start_time: datetime, end_time: datetime):
        self.start_time = start_time
        self.end_time = end_time
        self.evidence: List[Evidence] = []
    
    def add_evidence(self, evidence_type: str, source: str, data: Any, 
                    confidence: float = 1.0, timestamp: datetime = None) -> None:
        """Add evidence to the chain"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        evidence_item = Evidence(
            evidence_type=evidence_type,
            source=source,
            data=data,
            timestamp=timestamp,
            confidence=confidence
        )
        
        self.evidence.append(evidence_item)
    
    def get_evidence_by_type(self, evidence_type: str) -> List[Evidence]:
        """Get all evidence of a specific type"""
        return [e for e in self.evidence if e.evidence_type == evidence_type]
    
    def get_evidence_summary(self) -> Dict[str, Any]:
        """Get a summary of all evidence"""
        
        summary = {
            'total_evidence': len(self.evidence),
            'types': {},
            'confidence_avg': 0.0,
            'time_range': f"{self.start_time} ~ {self.end_time}"
        }
        
        if self.evidence:
            # Count by type
            for evidence in self.evidence:
                evidence_type = evidence.evidence_type
                if evidence_type not in summary['types']:
                    summary['types'][evidence_type] = 0
                summary['types'][evidence_type] += 1
            
            # Average confidence
            total_confidence = sum(e.confidence for e in self.evidence)
            summary['confidence_avg'] = total_confidence / len(self.evidence)
        
        return summary
