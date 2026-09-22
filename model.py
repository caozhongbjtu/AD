import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleCrossAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True)

    def forward(self,t,f):
        t2f,_=self.attn(t,f,f)
        f2t,_=self.attn(f,t,t)
        return t+t2f+f2t

class MultiScaleAutoencoder(nn.Module):
    def __init__(self,seq_len,beta,scales=3,encoding_dim=128):
        super().__init__()
        self.seq_len=seq_len
        self.scales=[2**i for i in range(scales)]
        self.encoding_dim=encoding_dim
        self.beta=beta

        self.t_encoders=nn.ModuleList()
        self.f_amp_encoders=nn.ModuleList()
        self.f_phase_encoders=nn.ModuleList()

        for s in self.scales:
            seg_len=seq_len//s
            freq_dim=seg_len//2+1
            self.t_encoders.append(nn.Sequential(
                nn.Linear(seg_len,encoding_dim*2),
                nn.ReLU(),
                nn.Linear(encoding_dim*2,encoding_dim)
            ))
            self.f_amp_encoders.append(nn.Sequential(
                nn.Linear(freq_dim,encoding_dim*2),
                nn.ReLU(),
                nn.Linear(encoding_dim*2,encoding_dim)
            ))
            self.f_phase_encoders.append(nn.Sequential(
                nn.Linear(freq_dim*2,encoding_dim*2),
                nn.ReLU(),
                nn.Linear(encoding_dim*2,encoding_dim)
            ))

        self.cross_attn_amp=nn.ModuleList([ScaleCrossAttention(encoding_dim) for _ in self.scales])
        self.cross_attn_phase=nn.ModuleList([ScaleCrossAttention(encoding_dim) for _ in self.scales])
        self.phase_gate=nn.Parameter(torch.zeros(len(self.scales)))

        self.scale_mlp=nn.Sequential(
            nn.Linear(encoding_dim,encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim,1)
        )

        self.decoder=nn.Sequential(
            nn.Linear(encoding_dim,encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2,seq_len)
        )

        self.consistency_loss_fn=nn.MSELoss()

    def segment(self,x,num_segments):
        B,C,L=x.shape
        seg_len=L//num_segments
        return x[:,:,:seg_len*num_segments].reshape(B,C,num_segments,seg_len)

    def forward(self,x):
        B,C,L=x.shape
        scale_feats=[]
        consistency_losses=[]

        for i,s in enumerate(self.scales):
            x_seg=self.segment(x,s)
            f_t=self.t_encoders[i](x_seg)
                
            xf=torch.fft.rfft(x_seg,dim=-1,norm="ortho")
            amp=torch.abs(xf)
            phase=torch.angle(xf)
            phase=torch.stack([torch.cos(phase),torch.sin(phase)],dim=-1)

            B_,C_,S_,F_=amp.shape
            amp=amp.reshape(B_*C_,S_,F_)
            phase=phase.reshape(B_*C_,S_,2*F_)
            f_t=f_t.reshape(B*C,S_,-1)
            f_amp=self.f_amp_encoders[i](amp)
            f_phase=self.f_phase_encoders[i](phase)
            

            fused_amp=self.cross_attn_amp[i](f_t,f_amp)
            fused_phase=self.cross_attn_phase[i](f_t,f_phase)
            fused=fused_amp+torch.sigmoid(self.phase_gate[i])*fused_phase

            t_mean = t.mean(dim=2)
            amp_mean = f_amp.mean(dim=2)
            phase_mean = f_phase.mean(dim=2)
            loss_amp = self.consistency_loss_fn(t_mean, amp_mean)
            loss_phase = self.consistency_loss_fn(t_mean, phase_mean)
            consistency_losses.append(loss_amp+self.beta*loss_phase)

            fused=F.avg_pool1d(fused.reshape(B,C,-1),kernel_size=s,stride=s)
            scale_feats.append(fused)

        stack=torch.stack(scale_feats,dim=2)
        weight=torch.softmax(self.scale_mlp(stack),dim=2)
        z=(stack*weight).sum(dim=2)
        recon=self.decoder(z)
        consistency_loss=sum(consistency_losses)/len(consistency_losses)

        return recon,None,None,z,None,consistency_loss
