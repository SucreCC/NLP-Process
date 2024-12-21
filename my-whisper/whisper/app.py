from model_dimensions import ModelDimensions

if __name__ == '__main__':

    # small whisper
    model_config = dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=512,
        n_audio_head=8,
        n_audio_layer=6,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=512,
        n_text_head=8,
        n_text_layer=6,
    )
