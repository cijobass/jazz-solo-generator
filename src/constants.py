"""
定数と設定値を管理するモジュール
"""

# ジャズノート（スケール）の定義
JAZZ_NOTES = {
    'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],  # Cメジャースケール
    'C7': ['C', 'E', 'G', 'Bb', 'D', 'F', 'A'],  # C7（ミクソリディアン+9,13）
    'Cm7': ['C', 'Eb', 'G', 'Bb', 'D', 'F'],  # Cマイナー7（ドリアン）
    'Cmaj7': ['C', 'D', 'E', 'G', 'A', 'B'],  # Cメジャー7（イオニアン）
    'C7alt': ['C', 'Db', 'Eb', 'E', 'Gb', 'G', 'A', 'Bb'],  # C7alt
    
    # ビバップスケール
    'C_bebop': ['C', 'D', 'E', 'F', 'G', 'Ab', 'A', 'B'],  # Cビバップメジャー
    'C7_bebop': ['C', 'D', 'E', 'F', 'G', 'A', 'Bb', 'B'],  # C7ビバップドミナント
    'Cm7_bebop': ['C', 'D', 'Eb', 'F', 'G', 'A', 'Bb', 'B'],  # Cmビバップマイナー
    
    # モーダルスケール
    'C_dorian': ['C', 'D', 'Eb', 'F', 'G', 'A', 'Bb'],  # Cドリアン
    'C_phrygian': ['C', 'Db', 'Eb', 'F', 'G', 'Ab', 'Bb'],  # Cフリジアン
    'C_lydian': ['C', 'D', 'E', 'F#', 'G', 'A', 'B'],  # Cリディアン
    'C_mixolydian': ['C', 'D', 'E', 'F', 'G', 'A', 'Bb'],  # Cミクソリディアン
    
    # フュージョンスケール
    'C_lydian_dominant': ['C', 'D', 'E', 'F#', 'G', 'A', 'Bb'],  # Cリディアンドミナント
    'C_altered': ['C', 'Db', 'Eb', 'E', 'Gb', 'Ab', 'Bb'],  # Cオルタード
    'C_diminished': ['C', 'D', 'Eb', 'F', 'Gb', 'Ab', 'A', 'B'],  # Cディミニッシュ
}

# 音名から半音数へのマッピング
NOTES_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
    'A#': 10, 'Bb': 10, 'B': 11
}

# 半音数から音名へのマッピング（逆引き）
INDEX_TO_NOTE = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'
}

# コード進行のプリセット
CHORD_PROGRESSIONS = {
    'II-V-I in C': 'Dm7,G7,Cmaj7',
    'Blues in F': 'F7,Bb7,F7,F7,Bb7,Bb7,F7,F7,C7,Bb7,F7,C7',
    'Autumn Leaves (最初の8小節)': 'Cm7,F7,BbM7,EbM7,Am7b5,D7,Gm7,Gm7',
    'Take the A Train (最初の8小節)': 'C6,C6,D7,D7,Dm7,G7,C6,C6',
    'All The Things You Are (最初の8小節)': 'Fm7,Bbm7,Eb7,AbM7,DbM7,Dm7b5,G7,CM7',
    'So What': 'Dm7,Dm7,Dm7,Dm7,Dm7,Dm7,Dm7,Dm7,EbM7,EbM7,EbM7,EbM7,Dm7,Dm7,Dm7,Dm7'
}

# ジャズスタイル別のパラメータ
JAZZ_STYLES = {
    'スタンダード': {
        'rhythm_complexity': 0.5,  # リズムの複雑さ (0.0-1.0)
        'chromatic_probability': 0.2,  # クロマチック音を使う確率
        'rest_probability': 0.2,  # レストを入れる確率
        'syncopation': 0.3,  # シンコペーションの強さ
        'scale_preference': 'normal',  # スケール選択の傾向
        'pattern_repetition': 0.3,  # パターン繰り返しの確率
        'description': 'バランスの取れた伝統的なジャズソロスタイル'
    },
    
    'ビバップ': {
        'rhythm_complexity': 0.8,  # 複雑なリズム
        'chromatic_probability': 0.6,  # 多くのクロマチック音
        'rest_probability': 0.1,  # 少ないレスト
        'syncopation': 0.7,  # 強いシンコペーション
        'scale_preference': 'bebop',  # ビバップスケール選好
        'pattern_repetition': 0.2,  # 少ないパターン繰り返し
        'description': 'チャーリー・パーカーやディジー・ガレスピースタイルの速く技巧的なフレーズ'
    },
    
    'モーダル': {
        'rhythm_complexity': 0.4,  # 控えめなリズム複雑さ
        'chromatic_probability': 0.1,  # 少ないクロマチック音
        'rest_probability': 0.3,  # 多めのレスト（スペース）
        'syncopation': 0.4,  # 中程度のシンコペーション
        'scale_preference': 'modal',  # モードスケール選好
        'pattern_repetition': 0.6,  # 多めのパターン繰り返し
        'description': 'マイルス・デイビス「Kind of Blue」風の落ち着いたモーダルアプローチ'
    },
    
    'フュージョン': {
        'rhythm_complexity': 0.7,  # 高めのリズム複雑さ
        'chromatic_probability': 0.4,  # 中程度のクロマチック音
        'rest_probability': 0.2,  # 標準的なレスト
        'syncopation': 0.8,  # 非常に強いシンコペーション
        'scale_preference': 'fusion',  # フュージョン系スケール選好
        'pattern_repetition': 0.4,  # 中程度のパターン繰り返し
        'description': 'ハービー・ハンコックやチック・コリア風の現代的で冒険的なサウンド'
    }
}

# MIDI/オーディオ関連の設定
SOUNDFONT_PATH = "C:/Program Files/fluidsynth/soundfonts/FluidR3_GM_GS.sf2"