"""
音声処理関連の機能を提供するモジュール
"""
import os
import base64
import tempfile
import streamlit as st
from midi2audio import FluidSynth
from src.constants import SOUNDFONT_PATH

def midi_to_audio(midi_file):
    """MIDIファイルをWAVに変換する関数
    
    Args:
        midi_file (str): MIDIファイルのパス
        
    Returns:
        str: 生成されたWAVファイルのパス、またはNone（エラー時）
    """
    try:
        # 一時ファイルを作成
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # サウンドフォントのパスを指定
        soundfont_path = SOUNDFONT_PATH
        
        # FluidSynthを使用してMIDIをWAVに変換
        fs = FluidSynth(sound_font=soundfont_path)
        fs.midi_to_audio(midi_file, temp_wav.name)
        
        return temp_wav.name
    except Exception as e:
        st.error(f"音声変換エラー: {e}")
        return None

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """ダウンロードリンクを生成する関数
    
    Args:
        bin_file (str): バイナリファイルのパス
        file_label (str): ダウンロードボタンに表示するラベル
        
    Returns:
        str: HTMLダウンロードリンク
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href