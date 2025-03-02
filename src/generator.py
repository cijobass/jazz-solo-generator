"""
ジャズソロ生成機能を提供するモジュール
"""
import random
from music21 import chord, note, stream, tempo
from src.constants import JAZZ_NOTES, NOTES_MAP, INDEX_TO_NOTE, JAZZ_STYLES
from src.theory import transpose_note, transpose_scale, parse_chord_symbol, get_chord_tones
from src.patterns import get_rhythm_patterns, create_repeated_pattern, vary_pattern

class JazzSoloGenerator:
    """ジャズソロを生成するクラス"""
    
    def __init__(self, style="スタンダード"):
        """ジャズソロジェネレーターの初期化
        
        Args:
            style (str): ジャズのスタイル（"スタンダード", "ビバップ", "モーダル", "フュージョン"）
        """
        # ベースのノート（JAZZでよく使うスケール音）
        self.jazz_notes = JAZZ_NOTES
        
        # 移調用の音階マップ
        self.notes_map = NOTES_MAP
        
        # 逆引き用の音階マップ
        self.index_to_note = INDEX_TO_NOTE
        
        # スタイル設定
        self.set_style(style)
    
    def set_style(self, style):
        """ジャズスタイルを設定する
        
        Args:
            style (str): ジャズのスタイル名
        """
        if style in JAZZ_STYLES:
            self.style = style
            self.style_params = JAZZ_STYLES[style]
        else:
            self.style = "スタンダード"
            self.style_params = JAZZ_STYLES["スタンダード"]
            
    def get_style_description(self):
        """現在のスタイルの説明を返す"""
        return self.style_params.get("description", "")
    
    def get_notes_for_chord(self, chord_symbol):
        """指定されたコード記号に基づいて使用可能なノートのリストを返す"""
        root, chord_type = parse_chord_symbol(chord_symbol)
        
        # スタイルに基づいてスケールの選択を変更
        scale_preference = self.style_params.get('scale_preference', 'normal')
        
        # コードタイプに対応するベーススケールを取得
        base_scale_type = 'C' + chord_type if chord_type else 'C'
        
        # スタイル別のスケール選択ロジック
        if scale_preference == 'bebop':
            # ビバップスタイル: ビバップスケールを優先
            if chord_type == '7':
                bebop_scale_type = 'C7_bebop'
                if bebop_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[bebop_scale_type]
                    return transpose_scale(base_scale, root)
            elif chord_type == 'm7':
                bebop_scale_type = 'Cm7_bebop'
                if bebop_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[bebop_scale_type]
                    return transpose_scale(base_scale, root)
            elif chord_type == 'maj7':
                bebop_scale_type = 'C_bebop'
                if bebop_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[bebop_scale_type]
                    return transpose_scale(base_scale, root)
        
        elif scale_preference == 'modal':
            # モーダルスタイル: モードスケールを優先
            if chord_type == '7':
                modal_scale_type = 'C_mixolydian'
                if modal_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[modal_scale_type]
                    return transpose_scale(base_scale, root)
            elif chord_type == 'm7':
                modal_scale_type = 'C_dorian'
                if modal_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[modal_scale_type]
                    return transpose_scale(base_scale, root)
            elif chord_type == 'maj7':
                modal_scale_type = 'C_lydian'
                if modal_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[modal_scale_type]
                    return transpose_scale(base_scale, root)
        
        elif scale_preference == 'fusion':
            # フュージョンスタイル: より現代的なスケールを優先
            if chord_type == '7':
                fusion_scale_type = 'C_lydian_dominant'
                if fusion_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[fusion_scale_type]
                    return transpose_scale(base_scale, root)
            elif chord_type == 'm7':
                fusion_scale_type = 'C_dorian'  # m7にはドリアンスケールが一般的
                if fusion_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[fusion_scale_type]
                    return transpose_scale(base_scale, root)
            elif chord_type == 'maj7':
                fusion_scale_type = 'C_lydian'  # maj7にはリディアンスケールが一般的
                if fusion_scale_type in self.jazz_notes:
                    base_scale = self.jazz_notes[fusion_scale_type]
                    return transpose_scale(base_scale, root)
        
        # デフォルトのスケール選択
        if base_scale_type in self.jazz_notes:
            base_scale = self.jazz_notes[base_scale_type]
            return transpose_scale(base_scale, root)
        else:
            # 対応するスケールがない場合はCメジャースケールを移調して返す
            return transpose_scale(self.jazz_notes['C'], root)
    
    def generate_phrase(self, chord_symbol, length=8, octave=4, is_bass=False):
        """指定されたコードに合うフレーズを生成する"""
        notes_to_use = self.get_notes_for_chord(chord_symbol)
        phrase = []
        
        # スタイルパラメータの取得
        rhythm_complexity = self.style_params.get('rhythm_complexity', 0.5)
        chromatic_probability = self.style_params.get('chromatic_probability', 0.2)
        rest_probability = self.style_params.get('rest_probability', 0.2)
        syncopation = self.style_params.get('syncopation', 0.3)
        pattern_repetition = self.style_params.get('pattern_repetition', 0.3)
        
        # ベースラインの場合はレストを少なくする
        if is_bass:
            rest_probability = min(0.1, rest_probability)
            octave -= 1  # ベース用にオクターブを下げる
        
        # リズムパターンの取得
        rhythm_patterns = get_rhythm_patterns(is_bass, rhythm_complexity, syncopation)
        
        # ランダムにリズムパターンを選択
        selected_pattern = random.choice(rhythm_patterns)
        
        # フレーズの生成
        current_length = 0
        last_notes = []  # 過去数個の音符を記録
        
        # スタイルに基づいてパターン繰り返しを決定
        repeated_pattern = None
        if random.random() < pattern_repetition and not is_bass:
            # パターン繰り返しを使用
            pattern_length = random.choice([2, 3, 4])
            repeated_pattern = create_repeated_pattern(
                pattern_length, selected_pattern, notes_to_use, rest_probability, octave
            )
        
        # フレーズ生成のメインループ
        while current_length < length:
            # パターン繰り返しを使用する場合
            if repeated_pattern and random.random() < 0.7:
                pattern_total_length = 0
                # パターンの合計長を計算
                for item in repeated_pattern:
                    if isinstance(item[1], (int, float)):
                        pattern_total_length += item[1]
                
                # パターンが残りの長さに収まる場合
                if current_length + pattern_total_length <= length:
                    for pattern_item in repeated_pattern:
                        if pattern_item[0] == 'rest':
                            phrase.append(('rest', pattern_item[1]))
                        else:
                            note_name, pat_rhythm, pat_octave = pattern_item
                            phrase.append((note_name, pat_rhythm, pat_octave))
                            last_notes.append(note_name)
                        current_length += pattern_item[1]
                    # パターン後にバリエーションを加える
                    if random.random() < 0.3:
                        repeated_pattern = vary_pattern(repeated_pattern, notes_to_use)
                    continue
            
            # リズムパターンからランダムに選択
            duration = random.choice(selected_pattern)
            
            # パターンが残りの長さを超える場合は調整
            if current_length + duration > length:
                duration = length - current_length
            
            # 音符かレスト
            if random.random() < rest_probability:
                # レスト
                phrase.append(('rest', duration))
            else:
                # 音符
                if is_bass and current_length == 0:
                    # ベースラインの最初の音はルート音が自然
                    root, _ = parse_chord_symbol(chord_symbol)
                    note_name = root
                else:
                    # クロマチック音を追加するか決定
                    if random.random() < chromatic_probability and not is_bass:
                        # クロマチック音を追加（前の音に対して半音上/下）
                        if last_notes:
                            last_note = last_notes[-1]
                            if last_note in self.notes_map:
                                note_value = self.notes_map[last_note]
                                # 半音上か下にシフト
                                note_value = (note_value + random.choice([-1, 1])) % 12
                                note_name = self.index_to_note[note_value]
                            else:
                                note_name = random.choice(notes_to_use)
                        else:
                            note_name = random.choice(notes_to_use)
                    else:
                        # 通常のスケール音から選択
                        # コードトーン判定（C7の場合は C, E, G, Bb など）
                        root, chord_type = parse_chord_symbol(chord_symbol)
                        chord_tones = get_chord_tones(root, chord_type)
                        
                        # コードトーンを優先して選択
                        if chord_tones and random.random() < 0.6:
                            note_name = random.choice(chord_tones)
                        else:
                            note_name = random.choice(notes_to_use)
                
                # オクターブの処理
                actual_octave = octave
                
                # スタイルに基づいたオクターブ変化
                if not is_bass:
                    # 通常の確率でオクターブを上げる
                    if random.random() < 0.3:
                        actual_octave += 1
                    
                    # ビバップスタイルの場合、ワイドレンジで表現
                    if self.style == "ビバップ" and random.random() < 0.2:
                        actual_octave += random.choice([-1, 2])  # 下1オクターブか上2オクターブ
                    
                    # フュージョンスタイルの場合、より現代的な音域
                    if self.style == "フュージョン" and random.random() < 0.15:
                        actual_octave += 2  # 高音域を活用
                
                phrase.append((note_name, duration, actual_octave))
                last_notes.append(note_name)
                if len(last_notes) > 8:  # 過去の音符を記録する数を制限
                    last_notes.pop(0)
            
            current_length += duration
        
        return phrase
    
    def generate_solo(self, chord_progression, measures_per_chord=1, beats_per_measure=4, is_bass=False):
        """コード進行に基づいてソロを生成する"""
        solo = []
        
        for chord_symbol in chord_progression:
            # 各コードに対して指定した小節数のフレーズを生成
            phrase_length = measures_per_chord * beats_per_measure
            phrase = self.generate_phrase(chord_symbol, length=phrase_length, is_bass=is_bass)
            solo.extend(phrase)
        
        return solo
    
    def create_midi(self, solo, bass_line=None, tempo_value=120, output_file="jazz_solo.mid"):
        """生成したソロをMIDIファイルに変換"""
        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=tempo_value))
        
        # ソロパートを追加
        solo_part = stream.Part()
        for item in solo:
            if item[0] == 'rest':
                # レストの場合
                n = note.Rest()
                n.duration.quarterLength = item[1]
            else:
                # 音符の場合
                note_name, duration, octave = item
                n = note.Note(note_name + str(octave))
                n.duration.quarterLength = duration
            
            solo_part.append(n)
        
        s.append(solo_part)
        
        # ベースラインがある場合は追加
        if bass_line:
            bass_part = stream.Part()
            for item in bass_line:
                if item[0] == 'rest':
                    # レストの場合
                    n = note.Rest()
                    n.duration.quarterLength = item[1]
                else:
                    # 音符の場合
                    note_name, duration, octave = item
                    n = note.Note(note_name + str(octave))
                    n.duration.quarterLength = duration
                
                bass_part.append(n)
            
            s.append(bass_part)
        
        # MIDIファイルとして保存
        mf = s.write('midi', fp=output_file)
        return output_file