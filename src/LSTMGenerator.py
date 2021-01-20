import pickle
import torch
import nbimporter
from lstm_words import Encoder, Decoder

class LSTMGenerator():
        
    def __init__(self, president):
        # le president
        self.president = president
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load vocabularies
        self.xvoc = pickle.load(open('../data/lstm/models/{}_xvoc.pt'.format(president), 'rb'))
        self.yvoc = pickle.load(open('../data/lstm/models/{}_yvoc.pt'.format(president), 'rb'))
        
        # model parameters (fixed)
        self.hidden_size = 20
        self.embedding_size = 50
        self.num_layers = 2
        self.batch_size = 64
        self.window = 5
        
        # load models
        self.encoder = Encoder(len(self.xvoc), self.hidden_size, self.embedding_size, self.num_layers).to(self.device)
        self.encoder.load_state_dict(torch.load('../data/lstm/models/{}_enc.pt'.format(president))) #, map_location=torch.device('cpu')))
        self.decoder = Decoder(len(self.xvoc), self.hidden_size, self.embedding_size, self.num_layers).to(self.device)
        self.decoder.load_state_dict(torch.load('../data/lstm/models/{}_dec.pt'.format(president))) #, map_location=torch.device('cpu')))
    
    def voc_index(self, words):
        return torch.tensor([self.xvoc.stoi[x] for x in words]).to(self.device)
       
    def predict(self, inp, RND_FACTOR=0, multiply=False, h0=None, c0=None):
        with torch.no_grad():

            if h0 == None:
                h0 = torch.zeros(2*self.num_layers, self.batch_size, self.hidden_size).to(self.device)
            if c0 == None:
                c0 = torch.zeros(2*self.num_layers, self.batch_size, self.hidden_size).to(self.device)

            for w in range(inp.size(0)):
                    enc_out, (h0, c0) = self.encoder(inp[w], h0, c0)

            cur = inp[self.window-1].unsqueeze(0)
            dec_out, (h0, c0) = self.decoder(cur, h0, c0)

            # randomize
            if multiply:
                rnd = torch.rand(dec_out.shape).to(self.device) * RND_FACTOR + 1
                cur = torch.argmax(dec_out * rnd,dim=1)
            else:
                rnd = torch.rand(dec_out.shape).to(self.device) * RND_FACTOR
                cur = torch.argmax(dec_out.add(rnd),dim=1)

            return self.yvoc.itos[cur[0].item()], (h0, c0)

    def generate(self, intro=['good', 'evening', 'ladies', 'and', 'gentlemen'], multiply=True, rnd_factor=1.2, length=4000, carry=True):
        text = intro
        h0 = torch.zeros(2*self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(2*self.num_layers, self.batch_size, self.hidden_size).to(self.device)

        for i in range(length):
            cur_window = text[-self.window:]
            vecs = self.voc_index(cur_window).view(self.window,1).repeat(1,self.batch_size)

            if carry:
                prediction, (h0, c0) = self.predict(vecs, rnd_factor, multiply, h0, c0)
            else:
                prediction, _ = self.predict(vecs, rnd_factor, multiply)

            text.append(prediction)

        return ' '.join(text) 
    
    def generate_n(self, N=10, intro=['good', 'evening', 'ladies', 'and', 'gentlemen'], multiply=True, rnd_factor=1.2, length=4000, carry=True):
        generated = []

        for i in range(N):
            generated.append(
                self.generate(intro=['good', 'evening', 'ladies', 'and', 'gentlemen'], 
                         multiply=multiply, 
                         rnd_factor=rnd_factor, 
                         length=length, 
                         carry=carry)
            
            )
        return generated

    def persist(self, generated, path=None):
        if path == None: path = "../data/lstm/{}_generated/".format(self.president)
        
        # persist
        for i in range(len(generated)):
            with open("{}{}.txt".format(path, str(i)), "w") as text_file:
                text_file.write(generated[i])