"""
音楽理論関連の機能を提供するモジュール
"""
from src.constants import NOTES_MAP, INDEX_TO_NOTE

def transpose_note(note_name, root, notes_map=NOTES_MAP, index_to_note=INDEX_TO_NOTE):
    """ノートを指定されたルート音に移調する
    
    Args:
        note_name (str): 移調する音名
        root (str): 移調先のルート音
        notes_map (dict): 音名から半音数へのマッピング
        index_to_note (dict): 半音数から音名へのマッピング
        
    Returns:
        str: 移調後の音名
    """
    if note_name in notes_map and root in notes_map:
        # 元のノートの半音数とルートの半音数の差を計算
        transpose_interval = notes_map[root] - notes_map['C']
        original_value = notes_map[note_name]
        # 移調後の半音数を計算（12音でループさせる）
        new_value = (original_value + transpose_interval) % 12
        return index_to_note[new_value]
    return note_name  # マップにない場合はそのまま返す

def transpose_scale(scale, root, notes_map=NOTES_MAP, index_to_note=INDEX_TO_NOTE):
    """スケール全体を指定されたルート音に移調する
    
    Args:
        scale (list): 移調するスケール音のリスト
        root (str): 移調先のルート音
        notes_map (dict): 音名から半音数へのマッピング
        index_to_note (dict): 半音数から音名へのマッピング
        
    Returns:
        list: 移調後のスケール音のリスト
    """
    return [transpose_note(note, root, notes_map, index_to_note) for note in scale]

def parse_chord_symbol(chord_symbol):
    """コード記号を解析してルート音とタイプを取得する
    
    Args:
        chord_symbol (str): コード記号（例: "Cm7", "G7", "Fmaj7"）
        
    Returns:
        tuple: (root, chord_type)
    """
    if len(chord_symbol) == 0:
        return 'C', ''  # デフォルト
    
    # ルート音を判断（最初の1-2文字）
    if len(chord_symbol) > 1 and chord_symbol[1] in ['#', 'b']:
        root = chord_symbol[0:2]
        chord_type = chord_symbol[2:]
    else:
        root = chord_symbol[0]
        chord_type = chord_symbol[1:]
    
    # コードタイプをマッピング（簡易版）
    if chord_type == '7':
        return root, '7'
    elif chord_type == 'm7' or chord_type == '-7':
        return root, 'm7'
    elif chord_type == 'maj7' or chord_type == 'M7':
        return root, 'maj7'
    elif '7alt' in chord_type:
        return root, '7alt'
    else:
        # その他のコードタイプ（デフォルトでは7を返す）
        return root, '7'

def get_chord_tones(root, chord_type, notes_map=NOTES_MAP, index_to_note=INDEX_TO_NOTE):
    """指定されたルートとコードタイプに基づいてコードトーンを取得する
    
    Args:
        root (str): ルート音
        chord_type (str): コードタイプ（"7", "m7", "maj7"など）
        notes_map (dict): 音名から半音数へのマッピング
        index_to_note (dict): 半音数から音名へのマッピング
        
    Returns:
        list: コードトーンのリスト
    """
    chord_tones = []
    if root in notes_map:
        # ルート
        chord_tones.append(root)
        
        # 3rd（メジャーかマイナーか）
        if 'm' in chord_type:
            # マイナー3rd
            third_value = (notes_map[root] + 3) % 12
        else:
            # メジャー3rd
            third_value = (notes_map[root] + 4) % 12
        chord_tones.append(index_to_note[third_value])
        
        # 5th
        fifth_value = (notes_map[root] + 7) % 12
        chord_tones.append(index_to_note[fifth_value])
        
        # 7th
        if 'maj7' in chord_type:
            # メジャー7th
            seventh_value = (notes_map[root] + 11) % 12
        else:
            # マイナー7th
            seventh_value = (notes_map[root] + 10) % 12
        chord_tones.append(index_to_note[seventh_value])
    
    return chord_tones