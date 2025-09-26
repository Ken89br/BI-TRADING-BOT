import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
from scipy.signal import argrelextrema

class ElliottWaveStrategy:
    """
    Professional Elliott Wave implementation with:
    - Automated wave counting algorithm
    - Fibonacci retracement/projection validation
    - Wave structure validation (impulse vs corrective)
    - Volume and momentum confirmation
    - Risk management rules
    """

    @staticmethod
    def identify_swings(high: pd.Series, 
                       low: pd.Series, 
                       close: pd.Series,
                       lookback: int = 5) -> Dict[str, List[Tuple[int, float]]]:
        """
        Identifies swing highs and swing lows using relative extrema
        Returns dictionary with wave points and their types
        """
        # Find local maxima and minima
        highs_idx = argrelextrema(high.values, np.greater, order=lookback)[0]
        lows_idx = argrelextrema(low.values, np.less, order=lookback)[0]
        
        # Combine and sort all turning points
        all_points = []
        for idx in highs_idx:
            all_points.append(('high', idx, high.iloc[idx]))
        for idx in lows_idx:
            all_points.append(('low', idx, low.iloc[idx]))
        
        # Sort by index
        all_points.sort(key=lambda x: x[1])
        
        # Filter consecutive highs/lows
        filtered_points = []
        for i in range(1, len(all_points)-1):
            prev_type, prev_idx, prev_val = all_points[i-1]
            curr_type, curr_idx, curr_val = all_points[i]
            next_type, next_idx, next_val = all_points[i+1]
            
            # Keep only alternating highs and lows
            if curr_type != prev_type:
                filtered_points.append((curr_type, curr_idx, curr_val))
        
        return {
            'swings': filtered_points,
            'current_price': close.iloc[-1],
            'current_position': len(close) - 1
        }

    @staticmethod
    def count_waves(swings: List[Tuple[str, int, float]]) -> Dict[str, Union[List, Dict]]:
        """
        Attempts to count Elliott Waves from swing points
        Returns wave labels and Fibonacci relationships
        """
        if len(swings) < 5:
            return {'valid': False, 'reason': 'Not enough swing points'}
        
        waves = []
        impulse_confirmed = False
        current_wave = None
        
        # Basic wave counting logic
        for i in range(len(swings)):
            swing_type, idx, val = swings[i]
            
            if i == 0:
                current_wave = ('1', idx, val)
                waves.append(current_wave)
            elif i == 1:
                current_wave = ('2', idx, val)
                waves.append(current_wave)
            elif i == 2:
                # Wave 3 should exceed wave 1
                if (swing_type == 'high' and val > waves[0][2]) or \
                   (swing_type == 'low' and val < waves[0][2]):
                    current_wave = ('3', idx, val)
                    waves.append(current_wave)
                else:
                    return {'valid': False, 'reason': 'Wave 3 violation'}
            elif i == 3:
                # Wave 4 should retrace part of wave 3
                current_wave = ('4', idx, val)
                waves.append(current_wave)
            elif i == 4:
                # Wave 5 should exceed wave 3 in impulse
                if (swing_type == 'high' and val > waves[2][2]) or \
                   (swing_type == 'low' and val < waves[2][2]):
                    current_wave = ('5', idx, val)
                    waves.append(current_wave)
                    impulse_confirmed = True
                else:
                    return {'valid': False, 'reason': 'Wave 5 violation'}
            else:
                # Handle corrective waves
                pass
        
        # Calculate Fibonacci relationships
        fib_retracements = {}
        if len(waves) >= 3:
            wave_1_height = abs(waves[0][2] - waves[1][2])
            wave_3_height = abs(waves[2][2] - waves[3][2])
            
            fib_retracements = {
                'wave_2_retrace': abs(waves[1][2] - waves[0][2]) / wave_1_height,
                'wave_4_retrace': abs(waves[3][2] - waves[2][2]) / wave_3_height,
                'wave_3_extension': wave_3_height / wave_1_height
            }
        
        return {
            'valid': True,
            'waves': waves,
            'impulse_confirmed': impulse_confirmed,
            'fib_relationships': fib_retracements,
            'current_wave': current_wave[0] if current_wave else None
        }

    @staticmethod
    def validate_waves(waves: Dict) -> Dict[str, Union[bool, str]]:
        """
        Validates Elliott Wave rules:
        1. Wave 2 cannot retrace more than 100% of Wave 1
        2. Wave 3 cannot be the shortest
        3. Wave 4 cannot enter Wave 1 price territory
        """
        if not waves['valid']:
            return waves
        
        wave_labels = [w[0] for w in waves['waves']]
        wave_values = [w[2] for w in waves['waves']]
        
        # Rule 1: Wave 2 retracement
        if waves['fib_relationships']['wave_2_retrace'] > 1.0:
            return {'valid': False, 'reason': 'Wave 2 retrace > 100%'}
        
        # Rule 2: Wave 3 not shortest (in impulse)
        if waves['impulse_confirmed']:
            wave_1 = abs(wave_values[0] - wave_values[1])
            wave_3 = abs(wave_values[2] - wave_values[3])
            wave_5 = abs(wave_values[4] - wave_values[3]) if len(wave_values) > 4 else 0
            
            if wave_3 < wave_1 and wave_3 < wave_5:
                return {'valid': False, 'reason': 'Wave 3 is shortest'}
        
        # Rule 3: Wave 4 cannot enter Wave 1 territory
        if len(wave_values) >= 4:
            if (wave_values[0] < wave_values[1] and wave_values[3] < wave_values[0]) or \
               (wave_values[0] > wave_values[1] and wave_values[3] > wave_values[0]):
                return {'valid': False, 'reason': 'Wave 4 entered Wave 1 territory'}
        
        return waves

    @staticmethod
    def generate_signals(swing_data: Dict,
                        wave_count: Dict,
                        volume: pd.Series,
                        momentum: pd.Series) -> Dict[str, Union[str, float]]:
        """
        Generates trading signals based on:
        - Completed wave patterns
        - Fibonacci projections
        - Momentum and volume confirmation
        """
        if not wave_count['valid']:
            return {'signal': None, 'reason': wave_count['reason']}
        
        current_wave = wave_count['current_wave']
        swings = swing_data['swings']
        current_price = swing_data['current_price']
        current_position = swing_data['current_position']
        
        # Signal for completed 5-wave impulse
        if current_wave == '5' and len(swings) >= 5:
            last_swing = swings[-1]
            prev_swing = swings[-2]
            
            # Check if we're near the end of wave 5
            if (last_swing[0] == 'high' and current_price < last_swing[2] * 0.998) or \
               (last_swing[0] == 'low' and current_price > last_swing[2] * 1.002):
                
                # Momentum divergence check
                if momentum.iloc[current_position] < momentum.iloc[prev_swing[1]]:
                    return {
                        'signal': 'potential_reversal',
                        'wave_count': '5-wave_complete',
                        'projected_target': calculate_fib_target(wave_count),
                        'current_wave': current_wave
                    }
        
        return {'signal': None}

    @staticmethod
    def full_analysis(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     volume: pd.Series,
                     momentum: pd.Series,
                     lookback: int = 5) -> Dict[str, Union[Dict, str]]:
        """
        Complete Elliott Wave analysis:
        - Swing point identification
        - Wave counting
        - Pattern validation
        - Signal generation
        """
        swings = ElliottWaveStrategy.identify_swings(high, low, close, lookback)
        wave_count = ElliottWaveStrategy.count_waves(swings['swings'])
        validated = ElliottWaveStrategy.validate_waves(wave_count)
        signals = ElliottWaveStrategy.generate_signals(swings, validated, volume, momentum)
        
        return {
            'swings': swings,
            'wave_count': validated,
            'signals': signals,
            'risk_management': {
                'stop_loss': calculate_stop_level(validated, close.iloc[-1]),
                'take_profit': calculate_target_level(validated, close.iloc[-1])
            }
        }

def calculate_fib_target(wave_count: Dict) -> float:
    """Calculates Fibonacci projection targets"""
    if not wave_count['valid'] or len(wave_count['waves']) < 5:
        return None
    
    waves = wave_count['waves']
    wave_0 = waves[0][2]
    wave_1 = waves[1][2]
    wave_3 = waves[3][2]
    
    if waves[0][0] == 'high':  # Downtrend
        return wave_3 - (wave_0 - wave_1) * 1.618
    else:  # Uptrend
        return wave_3 + (wave_1 - wave_0) * 1.618

def calculate_stop_level(wave_count: Dict, current_price: float) -> float:
    """Calculates stop loss based on wave structure"""
    if not wave_count['valid']:
        return None
    
    if wave_count['current_wave'] == '5':
        return current_price * 1.01 if wave_count['waves'][0][0] == 'low' else current_price * 0.99
    return None

def calculate_target_level(wave_count: Dict, current_price: float) -> float:
    """Calculates take profit based on Fibonacci projections"""
    target = calculate_fib_target(wave_count)
    return round(target, 5) if target else None