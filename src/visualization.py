"""
楽譜表示などの視覚化機能を提供するモジュール
"""
import os
import tempfile
import matplotlib.pyplot as plt
import streamlit as st
from music21 import converter

def display_score(midi_file, max_measures=8):
    """MIDIファイルから楽譜を表示する関数
    
    Args:
        midi_file (str): MIDIファイルのパス
        max_measures (int): 表示する最大小節数
        
    Returns:
        str: 生成された楽譜画像のパス、またはNone（エラー時）
    """
    try:
        # MIDIファイルを読み込む
        score = converter.parse(midi_file)
        
        # パートを取得（通常は最初のパート）
        if len(score.parts) > 0:
            melody_part = score.parts[0]
        else:
            melody_part = score
        
        # 小節数を制限（大きすぎると表示が難しい）
        measures = melody_part.measures(0, max_measures)
        
        # 一時ファイルを作成して楽譜を画像として保存
        temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_img.close()
        
        # LilyPondがインストールされている場合はそちらを使用
        try:
            measures.write('musicxml.png', fp=temp_img.name)
        except:
            # LilyPondがない場合はmatplotlibで簡易表示
            plt.figure(figsize=(12, 6))
            plot = measures.plot()
            plt.savefig(temp_img.name)
            plt.close()
        
        return temp_img.name
    except Exception as e:
        st.error(f"楽譜表示エラー: {e}")
        return None