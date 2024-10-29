# local-simple-realtime-api

## 注意

- 本リポジトリは開発途中です。
- 現段階でも [LocalChatVRM](https://github.com/nyosegawa/LocalChatVRM) と連携し音声対話をすることができますが、llm/stream等、未実装のものも入っています。

## 概要

PC1台で動くシンプルなSpeech-to-Speechサーバーです。OpenAIのようにend-to-endなモデルではなく、音声を受け取り、ASRモデルでテキストにし、テキストから返答をLLMで生成し、LLMで生成したテキストをTTSで音声合成しています。また、個別にAPIサーバーとしても利用できます。

## コンセプト

見通しがよくシンプルな Speech-to-Speech サーバーとして作っています。

- シンプルな Speech-to-Speech サーバー
    - local で動くことを企図しているため、Web API対応等は行わない
- シンプルな対話
    - 相槌等の早期リターンは行わない

## Speech to Speech の処理の流れ

- Websocketサーバーでの音声受信
    - VADでユーザーの発話を認識します
    - ユーザーの発話後200msでターン交代とし、以下の音声認識等の処理を行います
- Speech to Text (音声認識)
    - Whisper の [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) モデルで音声認識をします
        - turboモデルは809 Mパラメータのため比較的軽量となっています。英語の場合、tinyやbaseでもよいでしょう。
- Text Generation (返答生成)
    - [google/gemma-2-2b-jpn-it](https://huggingface.co/google/gemma-2-2b-jpn-it) で返答生成をします
- Text to Speech (音声合成)
    - [Style-Bert-Vits2](https://github.com/litagin02/Style-Bert-VITS2) を使い、[litagin/style_bert_vits2_jvnv](https://huggingface.co/litagin/style_bert_vits2_jvnv) の jvnv-F1-jp/jvnv-F1-jp_e160_s14000 モデルで音声合成をします

## 実行

必要なライブラリ等のインストール

```
pip install -r requirements.txt
```

実行: run.bat をダブルクリック

## 要求スペック

LLMを量子化等せずそのまま載せているのでめちゃくちゃVRAMを食います。全部で7GB VRAMくらい食います。

- 8GB VRAM 以上
    - RTX 2080で動作確認済

## 高速化について

各サーバーをGCP等のインスタンスに配置するだけで早くなります。シンプルさのため対応しませんが蒸留モデルを使う等で推論速度を上げる他、LLMからstream出力し有効な発話単位で分割し逐次音声合成し音声パケットを送ることで更に短縮されます。適当にやっても1.6秒くらいになります

## Speech-to-Speechモデル

副産物としてASR、LLM、TTSの各サーバーを作れる嬉しさがあり、今回は旧来的なアプローチにしました。end-to-endなモデルを試す場合は [mini-omni2](https://github.com/gpt-omni/mini-omni2) 等を使うと良いと思います。私もtrainingしていますが、stage 2 の modality alignment がうまくいっていません

## 同様のプロジェクトの紹介

- [ggerganov/whisper.cpp/talk-llama](https://github.com/ggerganov/whisper.cpp/tree/master/examples/talk-llama)
    - より軽量なSTSを探している場合はこちらをおすすめします。
    - llama.cpp と whisper.cpp と 軽量 tts (edge-tts ライブラリのようなもの, MacのsayコマンドとWinのSpeechSynthesizerに対応) で構築されています
- [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
    - よりちゃんとした実装を探している場合はこちらをおすすめします。