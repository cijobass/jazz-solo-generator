"""
リズムパターンと生成パターンの機能を提供するモジュール
"""
import random

# ベースライン向けのリズムパターン
BASS_PATTERNS = [
    [1, 1, 1, 1],  # 基本的なウォーキングベース
    [1.5, 0.5, 1, 1],  # 付点8分音符+16分音符のパターン
    [1, 0.5, 0.5, 1, 1],  # シンコペーションパターン
]

# 基本的なソロラインパターン
BASIC_PATTERNS = [
    [1, 0.5, 0.5, 1, 1],  # クォーター、8分、8分、クォーター、クォーター
    [0.5, 0.5, 0.5, 0.5, 1, 1],  # 8分4つ、クォーター2つ
    [1, 1, 0.5, 0.5, 0.5, 0.5],  # クォーター2つ、8分4つ
]

# 複雑なソロラインパターン
COMPLEX_PATTERNS = [
    [0.33, 0.33, 0.33, 1, 1],  # トリプレット、クォーター2つ
    [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1],  # 16分4つ、8分2つ、クォーター
    [0.75, 0.25, 0.5, 0.5, 1, 1],  # 付点8分＋16分、8分2つ、クォーター2つ
    [0.25, 0.75, 0.5, 0.5, 0.25, 0.75],  # 16分＋付点8分、8分2つ、16分＋付点8分
]

# 非常に複雑なソロラインパターン
VERY_COMPLEX_PATTERNS = [
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],  # 16分8つ
    [0.33, 0.33, 0.33, 0.33, 0.33, 0.33],  # トリプレット6つ
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5],  # クインタプレット、8分2つ
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.5, 0.5],  # 32分8つ、8分2つ
]

# シンコペーションパターン
SYNCOPATED_PATTERNS = [
    [0.5, 1.5, 1, 1],  # シンコペーション例1
    [1.5, 0.5, 1, 1],  # シンコペーション例2
    [1, 1.5, 0.5, 1],  # シンコペーション例3
]

def get_rhythm_patterns(is_bass, rhythm_complexity, syncopation):
    """スタイルパラメータに基づいてリズムパターンを取得する
    
    Args:
        is_bass (bool): ベースラインかどうか
        rhythm_complexity (float): リズムの複雑さ (0.0-1.0)
        syncopation (float): シンコペーションの強さ (0.0-1.0)
        
    Returns:
        list: リズムパターンのリスト
    """
    if is_bass:
        return BASS_PATTERNS
    
    # ソロラインの場合はリズムの複雑さに応じてパターンを選択
    patterns = BASIC_PATTERNS.copy()
    
    if rhythm_complexity > 0.3:
        patterns.extend(COMPLEX_PATTERNS)
    
    if rhythm_complexity > 0.6:
        patterns.extend(VERY_COMPLEX_PATTERNS)
    
    # シンコペーションを追加
    if syncopation > 0.5:
        patterns.extend(SYNCOPATED_PATTERNS)
    
    return patterns

def create_repeated_pattern(length, rhythm_pattern, notes, rest_probability, octave):
    """繰り返し用のパターンを作成する
    
    Args:
        length (int): パターン長さ
        rhythm_pattern (list): リズムパターン
        notes (list): 使用可能な音符リスト
        rest_probability (float): レストを入れる確率
        octave (int): 基本オクターブ
        
    Returns:
        list: 生成されたパターン
    """
    pattern = []
    for _ in range(length):
        # リズムパターンから個別の値をランダムに選択
        rhythm_value = random.choice(rhythm_pattern)
        if random.random() < rest_probability:
            pattern.append(('rest', rhythm_value))
        else:
            note_name = random.choice(notes)
            pattern.append((note_name, rhythm_value, octave))
    return pattern

def vary_pattern(pattern, available_notes):
    """パターンにバリエーションを加える
    
    Args:
        pattern (list): 元のパターン
        available_notes (list): 使用可能な音符リスト
        
    Returns:
        list: バリエーションを加えたパターン
    """
    if not pattern:
        return pattern
        
    result = pattern.copy()
    # いくつかの音符を変更
    for i in range(len(result)):
        if result[i][0] != 'rest' and random.random() < 0.3:
            note_name, duration, oct = result[i]
            # 新しい音符を選択
            new_note = random.choice(available_notes)
            result[i] = (new_note, duration, oct)
    return result