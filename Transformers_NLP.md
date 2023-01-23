# English to Spanish NLP using Transformers
An implementation of a transformer model that learns the context by tracking relationships in sequential data like the words in the sentences of the English-Spanish dataset and achieved an accuarcy of over 75%.

### Transformers- Encoder and Decoder
The Transformer architecture follows an encoder-decoder structure but does not rely on recurrence and convolutions in order to generate an output. The task of the encoder is to map an input sequence to a sequence of continuous representations, which is then fed into a decoder. 
The decoder, on receiving the output of the encoder, combines it with the decoder output at the previous time step to generate an output sequence.
The Transformer architecture cannot inherently capture any information about the relative positions of the words in the sequence since it does not make use of recurrence. This information has to be injected by introducing positional encodings to the input embeddings. The positional encoding vectors are of the same dimension as the input embeddings and are generated using sine and cosine functions of different frequencies. They are simply summed to the input embeddings in order to inject the positional information.
The input to the decoder is also augmented by positional encoding in the same manner done on the encoder side.The output of the decoder finally passes through a fully connected layer, followed by a softmax layer, to generate a prediction for the next word of the output sequence. 

### Output
<img width="853" alt="nlp" src="https://user-images.githubusercontent.com/122699563/213961493-422b3e08-b939-4b63-a2b9-822687b5b076.png">

