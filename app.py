"""
ジャズアドリブジェネレーターのメインアプリケーション
"""
import streamlit as st
import time
import os
from src.generator import JazzSoloGenerator
from src.audio_utils import midi_to_audio, get_binary_file_downloader_html
from src.visualization import display_score
from src.constants import CHORD_PROGRESSIONS, JAZZ_STYLES

def main():
    """Streamlitアプリのメイン関数"""
    st.title('ジャズアドリブジェネレーター 🎷🎸')
    st.write('コード進行を入力して、ジャズソロを自動生成しましょう！')
    
    # サイドバーに説明を追加
    st.sidebar.title('使い方')
    st.sidebar.write('1. コード進行を入力（例: Dm7,G7,Cmaj7）')
    st.sidebar.write('2. テンポ、小節数などのパラメータを設定')
    st.sidebar.write('3. 「ソロを生成」ボタンをクリック')
    st.sidebar.write('4. 生成されたMIDIファイルをダウンロード')
    
    # プリセットのコード進行を選択するオプション
    preset_selection = st.selectbox(
        'プリセットコード進行を選択:',
        ['カスタム入力'] + list(CHORD_PROGRESSIONS.keys())
    )
    
    # コード進行の入力
    if preset_selection == 'カスタム入力':
        chord_input = st.text_input('コード進行をカンマ区切りで入力:', 'Dm7,G7,Cmaj7')
    else:
        chord_input = st.text_input('コード進行:', CHORD_PROGRESSIONS[preset_selection])
    
    # 各コードの小節数
    measures_per_chord = st.slider('各コードの小節数:', 1, 4, 1)
    
    # テンポ設定
    tempo = st.slider('テンポ (BPM):', 60, 300, 140)
    
    # オクターブ設定
    octave = st.slider('オクターブ:', 2, 6, 4)
    
    # 楽器選択（機能拡張）
    instrument_type = st.radio(
        "ライン生成タイプ:",
        ('ソロライン', 'ベースライン', '両方を生成')
    )
    
    # スタイル選択（機能拡張）
    style = st.selectbox(
        'スタイル:',
        ['スタンダード', 'ビバップ', 'モーダル', 'フュージョン']
    )
    
    # 選択したスタイルの説明を表示
    if style in JAZZ_STYLES:
        st.info(JAZZ_STYLES[style]['description'])
    
    # ボタンを押したときの処理
    if st.button('ソロを生成'):
        # 入力されたコード進行を分割
        chord_progression = [chord.strip() for chord in chord_input.split(',')]
        
        if not chord_progression:
            st.error('有効なコード進行を入力してください')
        else:
            # プログレスバーを表示
            progress_bar = st.progress(0)
            
            # JazzSoloGeneratorのインスタンスを作成
            generator = JazzSoloGenerator(style=style)
            
            # 選択に応じてソロとベースラインを生成
            is_bass = instrument_type == 'ベースライン'
            generate_both = instrument_type == '両方を生成'
            
            solo_line = None
            bass_line = None
            
            if is_bass or generate_both:
                st.write('ベースラインを生成中...')
                bass_line = generator.generate_solo(chord_progression, measures_per_chord, is_bass=True)
                progress_bar.progress(0.5)
            
            if not is_bass or generate_both:
                st.write('ソロラインを生成中...')
                solo_line = generator.generate_solo(chord_progression, measures_per_chord, is_bass=False)
                progress_bar.progress(0.75)
            
            # MIDIファイルのファイル名
            timestamp = int(time.time())
            if is_bass:
                output_file = f"jazz_bassline_{timestamp}.mid"
                midi_file = generator.create_midi(bass_line, tempo_value=tempo, output_file=output_file)
            elif generate_both:
                output_file = f"jazz_combo_{timestamp}.mid"
                midi_file = generator.create_midi(solo_line, bass_line, tempo_value=tempo, output_file=output_file)
            else:
                output_file = f"jazz_solo_{timestamp}.mid"
                midi_file = generator.create_midi(solo_line, tempo_value=tempo, output_file=output_file)
            
            progress_bar.progress(1.0)
            
            # 生成されたMIDIファイルのダウンロードリンクを表示
            st.success('生成完了！')
            st.markdown(get_binary_file_downloader_html(midi_file, 'MIDIファイルをダウンロード'), unsafe_allow_html=True)
            
            # コード進行の可視化（シンプルな表示）
            st.subheader('生成されたコード進行:')
            cols = st.columns(len(chord_progression))
            for i, chord in enumerate(chord_progression):
                cols[i].write(f"{chord}")
            
            # 楽譜表示
            st.subheader("生成されたソロの楽譜")
            with st.spinner("楽譜を生成中..."):
                score_image = display_score(midi_file)
                if score_image:
                    st.image(score_image)
                    # 一時ファイルを削除
                    try:
                        os.remove(score_image)
                    except:
                        pass
                else:
                    st.info("楽譜の表示に失敗しました。music21の設定を確認してください。")
            
            # 試聴機能
            try:
                # MIDIをWAVに変換して再生
                wav_file = midi_to_audio(midi_file)
                if wav_file:
                    st.subheader("生成されたソロを試聴")
                    with open(wav_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")
                    # 一時ファイルを削除
                    try:
                        os.remove(wav_file)
                    except:
                        pass
            except ModuleNotFoundError:
                st.warning("midi2audioモジュールがインストールされていません。")
                st.info("試聴機能を使用するには以下のコマンドを実行してください: pip install midi2audio")
            except FileNotFoundError:
                st.warning("FluidSynthまたはSoundFontファイルが見つかりませんでした。")
                st.info("試聴機能を使用するには、FluidSynthをインストールし、SoundFontファイルを正しい場所に配置してください。")
                st.code("サウンドフォントの場所: C:/Program Files/fluidsynth/soundfonts/FluidR3_GM_GS.sf2")
            except Exception as e:
                st.warning(f"試聴機能は現在利用できません: {e}")
                st.info("試聴機能を使用するには、FluidSynth と SoundFont ファイルのインストールが必要です。")
            
            # 注意書き
            st.info('MIDIファイルはDAWソフトやMIDIプレーヤーで開いて再生・編集できます。')

if __name__ == "__main__":
    main()