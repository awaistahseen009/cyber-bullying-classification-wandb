from keras.layers import LSTM , Bidirectional , Embedding, Dense , Input
from keras.models import Model

def build_model(embedding_dim , latent_dim , vocab_size , max_len, target_values):
    inp=Input(max_len, )
    emb=Embedding(vocab_size, embedding_dim , input_length=max_len)(inp)
    gru1=Bidirectional(LSTM(latent_dim, return_sequences=True))(emb)
    gru2=Bidirectional(LSTM(latent_dim))(gru1)
    out=Dense(target_values, activation='softmax')(gru2)
    model=Model(inp , out)
    return model