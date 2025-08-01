import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List
from dataclasses import dataclass, field
from tqdm import tqdm
import logging

@dataclass
class PWM:
    """Container for position weight matrix data."""
    name: str
    matrix: np.ndarray
    bases: List[str] = field(default_factory=lambda: ['A', 'C', 'G', 'T'])
    
    def get_consensus(self, prob_threshold: float = 0.25) -> str:
        """
        Get consensus sequence from PWM using IUPAC ambiguity codes.
        
        Args:
            prob_threshold: Probability threshold for including bases in ambiguity codes
            
        Returns:
            Consensus sequence with IUPAC codes
        """
        iupac_map = {
            'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
            'AC': 'M', 'AG': 'R', 'AT': 'W',
            'CG': 'S', 'CT': 'Y', 'GT': 'K',
            'ACG': 'V', 'ACT': 'H', 'AGT': 'D', 'CGT': 'B',
            'ACGT': 'N'
        }
        
        consensus = []
        for pos_probs in self.matrix.T:
            # Get bases above threshold
            significant_bases = ''.join(b for b, p in zip(self.bases, pos_probs) 
                                     if p >= prob_threshold)
            
            # Sort bases by probability
            significant_bases = ''.join(sorted(significant_bases))
            
            # Map to IUPAC code
            consensus.append(iupac_map.get(significant_bases, 'N'))
            
        return ''.join(consensus)

def iupac_match(a: str, b: str) -> bool:
    """Check if two IUPAC nucleotide codes match."""
    iupac = {
        'A': {'A'},
        'C': {'C'},
        'G': {'G'},
        'T': {'T'},
        'R': {'A', 'G'},
        'Y': {'C', 'T'},
        'S': {'G', 'C'},
        'W': {'A', 'T'},
        'K': {'G', 'T'},
        'M': {'A', 'C'},
        'B': {'C', 'G', 'T'},
        'D': {'A', 'G', 'T'},
        'H': {'A', 'C', 'T'},
        'V': {'A', 'C', 'G'},
        'N': {'A', 'C', 'G', 'T'}
    }
    
    a = a.upper()
    b = b.upper()
    
    if a not in iupac or b not in iupac:
        raise ValueError(f"Invalid IUPAC code: {a if a not in iupac else b}")
        
    return bool(iupac[a] & iupac[b])

def levenshtein_iupac(seq1: str, seq2: str) -> int:
    """Calculate Levenshtein distance between two DNA sequences with IUPAC codes."""
    if not seq1: return len(seq2)
    if not seq2: return len(seq1)
    
    # Initialize lists instead of range objects
    previous_row = list(range(len(seq2) + 1))
    current_row = [0] * (len(seq2) + 1)
    
    for i, c1 in enumerate(seq1):
        current_row[0] = i + 1
        
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (not iupac_match(c1, c2))
            
            current_row[j + 1] = min(insertions, deletions, substitutions)
            
        previous_row, current_row = current_row, [0] * (len(seq2) + 1)  # Reset current_row
        
    return previous_row[-1]

def setup_logger():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_jaspar(jaspar_file: str) -> PWM:
    """Parse a JASPAR format PWM file."""
    with open(jaspar_file) as f:
        lines = f.readlines()
    
    if not lines or len(lines) != 5:
        raise ValueError("Invalid JASPAR format")
        
    name = lines[0].split()[0]
    matrix = []
    
    for line in lines[1:]:
        nums = line.split('[')[1].split(']')[0].strip().split()
        matrix.append([float(x) for x in nums])
    
    matrix = np.array(matrix)
    matrix = matrix / matrix.sum(axis=0)
    
    return PWM(name=name, matrix=matrix)

def score_seqlet(pwm: PWM, seq: str) -> Tuple[float, int]:
    """
    Score a sequence against a PWM using IUPAC-aware Levenshtein distance.
    Returns normalized score (1 - distance/max_possible_distance) and start position.
    
    Args:
        pwm: PWM object containing the motif matrix
        seq: DNA sequence to score
        
    Returns:
        Tuple of (normalized score, best start position)
    """
    seq_len = len(seq)
    pwm_width = pwm.matrix.shape[1]
    consensus = pwm.get_consensus()
    
    # Handle sequences shorter than PWM
    if seq_len < pwm_width:
        max_score = float('-inf')
        best_pos = 0
        
        for i in range(pwm_width - seq_len + 1):
            cons_slice = consensus[i:i+seq_len]
            raw_dist = levenshtein_iupac(seq, cons_slice)
            norm_score = 1 - (raw_dist / max(len(seq), len(cons_slice)))
            
            if norm_score > max_score:
                max_score = norm_score
                best_pos = i
                
        return max_score, best_pos
        
    # Handle sequences same length as PWM
    elif seq_len == pwm_width:
        raw_dist = levenshtein_iupac(seq, consensus)
        norm_score = 1 - (raw_dist / len(consensus))
        return norm_score, 0
        
    # Handle sequences longer than PWM
    else:
        max_score = float('-inf')
        best_pos = 0
        
        for i in range(seq_len - pwm_width + 1):
            subseq = seq[i:i+pwm_width]
            raw_dist = levenshtein_iupac(subseq, consensus)
            norm_score = 1 - (raw_dist / len(consensus))
            
            if norm_score > max_score:
                max_score = norm_score
                best_pos = i
                
        return max_score, best_pos

def main():
    parser = argparse.ArgumentParser(description='Score seqlets against JASPAR PWM using IUPAC-aware Levenshtein distance')
    parser.add_argument('--jaspar', required=True, help='Path to JASPAR PWM file')
    parser.add_argument('--seqlets', required=True, help='Path to seqlets CSV file')
    parser.add_argument('--output', required=True, help='Path for output CSV')
    parser.add_argument('--min-score', type=float, default=0.0, 
                       help='Minimum score threshold')
    
    args = parser.parse_args()
    logger = setup_logger()
    
    # Load PWM
    logger.info(f"Loading PWM from {args.jaspar}")
    pwm = parse_jaspar(args.jaspar)
    logger.info(f"Consensus sequence: {pwm.get_consensus()}")
    
    # Load seqlets
    logger.info(f"Loading seqlets from {args.seqlets}")
    seqlets_df = pd.read_csv(args.seqlets)
    
    # Score seqlets
    logger.info("Scoring seqlets")
    scores = []
    positions = []
    for _, row in tqdm(seqlets_df.iterrows(), total=seqlets_df.shape[0]):
        score, pos = score_seqlet(pwm, row['sequence'])
        scores.append(score)
        positions.append(pos)
    
    # Add scores to dataframe
    seqlets_df['levenshtein_score'] = scores
    seqlets_df['levenshtein_position'] = positions
    seqlets_df['pwm_width'] = pwm.matrix.shape[1]
    
    # Filter and sort results
    results = seqlets_df[seqlets_df['levenshtein_score'] >= args.min_score].sort_values(
        'levenshtein_score', ascending=False
    )
    
    # Save results
    logger.info(f"Saving {len(results)} matches to {args.output}")
    results.to_csv(args.output, index=False)
    
    # Print summary statistics
    logger.info(f"Summary statistics:")
    logger.info(f"Mean score: {np.mean(scores):.3f}")
    logger.info(f"Max score: {np.max(scores):.3f}")
    logger.info(f"Number of matches above threshold: {len(results)}")

if __name__ == '__main__':
    main()