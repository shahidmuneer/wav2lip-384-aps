import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.transformer_encoder import TransformerEncoder
from models.conv import Conv2d, Conv3d


class SyncTransformer(nn.Module):
    def __init__(self, d_model=512):
        super(SyncTransformer, self).__init__()
        self.d_model = d_model
        layers = [16,32, 64, 128, 256, 512, 1024]
        self.vid_prenet = nn.Sequential(
            Conv3d(3, layers[0], kernel_size=7, stride=1, padding=3),

            Conv3d(layers[0], layers[1], kernel_size=5, stride=(1, 2, 1), padding=(1, 1, 2)),
            Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[1], layers[2], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[2], layers[3], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[3], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[4], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=(0, 0, 1)),
            Conv3d(layers[4], layers[4], kernel_size=1, stride=1, padding=0),
            
            
            ###################
            # Modified blocks
            ##################
            Conv3d(layers[4],layers[5] , kernel_size=3, stride=2, padding=1),
            Conv3d(layers[5], layers[5], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[5], layers[5], kernel_size=3, stride=1, padding=1, residual=True),  # 12, 12
            nn.BatchNorm3d(layers[5]),

            Conv3d(layers[5], layers[6], kernel_size=3, stride=2, padding=1),
            Conv3d(layers[6], layers[6], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[6], layers[6], kernel_size=3, stride=1, padding=1, residual=True),  # 6, 6
            nn.BatchNorm3d(layers[6]),

            Conv3d(layers[6], layers[6], kernel_size=3, stride=2, padding=1), # 3, 3
            Conv3d(layers[6], layers[6], kernel_size=3, stride=1, padding=0),
            Conv3d(layers[6], layers[6], kernel_size=1, stride=1, padding=0)) # 1, 1
        
        
        
        self.aud_prenet = nn.Sequential(
            Conv2d(1, layers[0], kernel_size=3, stride=1, padding=1),
            Conv2d(layers[0], layers[0], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[0], layers[0], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[0], layers[1], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[1], layers[2], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[2], layers[3], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(layers[3], layers[4], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),
            
            
            Conv2d(layers[4], layers[5], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[5], layers[5], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[5], layers[5], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[5], layers[6], kernel_size=3, stride=1, padding=(0, 1)),
            Conv2d(layers[6], layers[6], kernel_size=1, stride=1, padding=0), )

        self.audio_video_transformer = TransformerEncoder(embed_dim=d_model,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)
        self.video_audio_transformer = TransformerEncoder(embed_dim=d_model,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)
        self.cross_transformer = TransformerEncoder(embed_dim=d_model,
                                                  num_heads=8,
                                                  layers=4,
                                                  attn_dropout=0.0,
                                                  relu_dropout=0.1,
                                                  res_dropout=0.1,
                                                  embed_dropout=0.25,
                                                  attn_mask=True)
        # self.dropout = 
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, frame_seq, mel_seq):
        B = frame_seq.shape[0]
        # print(frame_seq.shape)
        # print(frame_seq.view(B, -1, 3, 48, 384).shape)
        vid_embedding = self.vid_prenet(frame_seq.view(B,-1,3,48,384).permute(0,2,3,4,1).contiguous())
        aud_embedding = self.aud_prenet(mel_seq)

        vid_embedding = vid_embedding.squeeze(2).squeeze(2)
        aud_embedding = aud_embedding.squeeze(2)

        vid_embedding = vid_embedding.permute(2, 0, 1).contiguous()
        aud_embedding = aud_embedding.permute(2, 0, 1).contiguous()

        av_embedding = self.audio_video_transformer(aud_embedding, vid_embedding, vid_embedding)
        va_embedding = self.video_audio_transformer(vid_embedding, aud_embedding, aud_embedding)

        tranformer_out = self.cross_transformer(av_embedding, va_embedding, va_embedding)
        t = av_embedding.shape[0]

        out = F.max_pool1d(tranformer_out.permute(1, 2, 0).contiguous(), t).squeeze(-1)
        h_pooled = self.fc(out)  # [batch_size, d_model]
        h_pooled = F.dropout(h_pooled, p=0.1, training=self.training)
        h_pooled = self.activ1(h_pooled)
        logits_clsf = (self.classifier(h_pooled))
        return logits_clsf.squeeze(-1)


"""
# Test Model
if __name__ == "__main__":
    mel_seq = torch.rand([4, 1, 80, 80])
    frame_seq = torch.rand([4, 75, 48, 96])
    model = SyncTransformer()
    output = model(frame_seq, mel_seq)
"""