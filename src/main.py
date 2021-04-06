import array
import math
import os
import random



def malloc(sz):
	return [None for _ in range(0,sz)]
def realloc(a,sz):
	if (a is None):
		return malloc(sz)
	assert sz==len(a)+1
	return a+[None]
def free(a):
	assert a!=None
def fread(o,sz,c,f):
	assert sz==float
	bf=array.array("f")
	bf.frombytes(f.read(c*4))
	for i in range(0,c):
		o[i]=bf[i]
	return c



def tanh(x):
	return (math.exp(2*x)-1)/(math.exp(2*x)+1)
def tanh_d(x):
	return 1-x*x
def sigmoid(x):
	return 1/(1+math.exp(-x))
def sigmoid_d(x):
	return (1-x)*x



class FullyConnectedLayer:
	def __init__(self,x,y,f):
		self.x=x
		self.y=y
		self.b=malloc(self.y)
		self.w=malloc(self.y)
		if (f!=None):
			assert fread(self.b,float,self.y,f)==self.y
			for i in range(0,self.y):
				self.w[i]=malloc(self.x)
				assert fread(self.w[i],float,self.x,f)==self.x
		else:
			for i in range(0,self.y):
				self.b[i]=0
				self.w[i]=malloc(self.x)
				for j in range(0,self.x):
					self.w[i][j]=random.random()*2-1



	def forward(self,in_):
		o=malloc(self.y)
		for i in range(0,self.y):
			o[i]=self.b[i]
			for j in range(0,self.x):
				o[i]+=self.w[i][j]*in_[j]
		return o



	def train(self,in_,tg,lr):
		o=malloc(self.x)
		for i in range(0,self.y):
			p=self.b[i]
			for j in range(0,self.x):
				p+=self.w[i][j]*in_[j]
			bg=p-tg[i]
			self.b[i]-=bg*lr
			for j in range(0,self.x):
				if (i==0):
					o[j]=0
				o[j]+=self.w[i][j]*bg
				self.w[i][j]-=bg*in_[j]*lr
		return o



	def save(self,f):
		a=array.array("f")
		a.fromlist(self.b)
		f.write(a.tobytes())
		for i in range(0,self.y):
			a=array.array("f")
			a.fromlist(self.w[i])
			f.write(a.tobytes())



class LSTMLayer:
	def __init__(self,x,y,f):
		self.x=x
		self.y=y
		self.bx=malloc(self.y)
		self.bf=malloc(self.y)
		self.bi=malloc(self.y)
		self.bo=malloc(self.y)
		self.wx=malloc(self.y)
		self.wf=malloc(self.y)
		self.wi=malloc(self.y)
		self.wo=malloc(self.y)
		self._sz=-1
		self._cl=None
		self._xhl=None
		self._cal=None
		self._fl=None
		self._il=None
		self._ol=None
		self._outl=None
		self._c=malloc(self.y)
		self._h=malloc(self.y)
		self._hg=None
		self._cg=None
		self._bxg=None
		self._bfg=None
		self._big=None
		self._bog=None
		self._wxg=None
		self._wfg=None
		self._wig=None
		self._wog=None
		if (f!=None):
			assert fread(self.bx,float,self.y,f)==self.y
			assert fread(self.bf,float,self.y,f)==self.y
			assert fread(self.bi,float,self.y,f)==self.y
			assert fread(self.bo,float,self.y,f)==self.y
			for i in range(0,self.y):
				self._c[i]=0
				self._h[i]=0
				self.wx[i]=malloc(self.x+self.y)
				self.wf[i]=malloc(self.x+self.y)
				self.wi[i]=malloc(self.x+self.y)
				self.wo[i]=malloc(self.x+self.y)
				assert fread(self.wx[i],float,self.x+self.y,f)==self.x+self.y
				assert fread(self.wf[i],float,self.x+self.y,f)==self.x+self.y
				assert fread(self.wi[i],float,self.x+self.y,f)==self.x+self.y
				assert fread(self.wo[i],float,self.x+self.y,f)==self.x+self.y
		else:
			for i in range(0,self.y):
				self._c[i]=0
				self._h[i]=0
				self.bx[i]=random.random()*0.2-0.1
				self.bf[i]=random.random()*0.2-0.1
				self.bi[i]=random.random()*0.2-0.1
				self.bo[i]=random.random()*0.2-0.1
				self.wx[i]=malloc(self.x+self.y)
				self.wf[i]=malloc(self.x+self.y)
				self.wi[i]=malloc(self.x+self.y)
				self.wo[i]=malloc(self.x+self.y)
				for j in range(0,self.x+self.y):
					self.wx[i][j]=random.random()*0.2-0.1
					self.wf[i][j]=random.random()*0.2-0.1
					self.wi[i][j]=random.random()*0.2-0.1
					self.wo[i][j]=random.random()*0.2-0.1



	def forward(self,in_,tr=False):
		if (tr==True and self._sz==-1):
			self._sz=0
			self._hg=malloc(self.y)
			self._cg=malloc(self.y)
			self._bxg=malloc(self.y)
			self._bfg=malloc(self.y)
			self._big=malloc(self.y)
			self._bog=malloc(self.y)
			self._wxg=malloc(self.y)
			self._wfg=malloc(self.y)
			self._wig=malloc(self.y)
			self._wog=malloc(self.y)
			for i in range(0,self.y):
				self._hg[i]=0
				self._cg[i]=0
				self._bxg[i]=0
				self._bfg[i]=0
				self._big[i]=0
				self._bog[i]=0
				self._wxg[i]=malloc(self.x+self.y)
				self._wfg[i]=malloc(self.x+self.y)
				self._wig[i]=malloc(self.x+self.y)
				self._wog[i]=malloc(self.x+self.y)
				for j in range(0,self.x+self.y):
					self._wxg[i][j]=0
					self._wfg[i][j]=0
					self._wig[i][j]=0
					self._wog[i][j]=0
		xh=malloc(self.x+self.y)
		for i in range(0,self.x):
			xh[i]=in_[i]
		for i in range(0,self.y):
			xh[self.x+i]=self._h[i]
		if (tr==True):
			self._sz+=1
			self._cl=realloc(self._cl,self._sz)
			self._xhl=realloc(self._xhl,self._sz)
			self._cl[self._sz-1]=self._c
			self._xhl[self._sz-1]=xh
		ca=malloc(self.y)
		f=malloc(self.y)
		i_=malloc(self.y)
		o=malloc(self.y)
		out=malloc(self.y)
		for i in range(0,self.y):
			ca[i]=self.bx[i]
			f[i]=self.bf[i]
			i_[i]=self.bi[i]
			o[i]=self.bo[i]
			for j in range(0,self.x+self.y):
				ca[i]+=self.wx[i][j]*xh[j]
				f[i]+=self.wf[i][j]*xh[j]
				i_[i]+=self.wi[i][j]*xh[j]
				o[i]+=self.wo[i][j]*xh[j]
			ca[i]=tanh(ca[i])
			f[i]=sigmoid(f[i])
			i_[i]=sigmoid(i_[i])
			o[i]=sigmoid(o[i])
			self._c[i]=ca[i]*i_[i]+self._c[i]*f[i]
			out[i]=tanh(self._c[i])
			self._h[i]=out[i]*o[i]
		if (tr==True):
			self._cal=realloc(self._cal,self._sz)
			self._fl=realloc(self._fl,self._sz)
			self._il=realloc(self._il,self._sz)
			self._ol=realloc(self._ol,self._sz)
			self._outl=realloc(self._outl,self._sz)
			self._cal[self._sz-1]=ca
			self._fl[self._sz-1]=f
			self._il[self._sz-1]=i_
			self._ol[self._sz-1]=o
			self._outl[self._sz-1]=out
		return self._h



	def backward(self,tg):
		self._sz-=1
		c=self._cl[self._sz]
		xh=self._xhl[self._sz]
		ca=self._cal[self._sz]
		f=self._fl[self._sz]
		i_=self._il[self._sz]
		o=self._ol[self._sz]
		out=self._outl[self._sz]
		hg=malloc(self.y)
		og=malloc(self.x)
		for i in range(0,self.y):
			hg[i]=0
		for i in range(0,self.x):
			og[i]=0
		for i in range(0,self.y):
			tg[i]+=self._hg[i]
			self._cg[i]=tanh_d(out[i])*o[i]*tg[i]+self._cg[i]
			lfg=c[i]*self._cg[i]*sigmoid_d(f[i])
			self._cg[i]*=f[i]
			lxg=tanh_d(ca[i])*i_[i]*self._cg[i]
			lig=ca[i]*self._cg[i]*sigmoid_d(i_[i])
			log=out[i]*tg[i]*sigmoid_d(o[i])
			self._bxg[i]+=lxg
			self._big[i]+=lig
			self._bfg[i]+=lfg
			self._bog[i]+=log
			for j in range(0,self.x+self.y):
				if (j<self.x):
					og[j]+=self.wx[i][j]*lxg+self.wi[i][j]*lig+self.wf[i][j]*lfg+self.wo[i][j]*log
				else:
					hg[j-self.x]+=self.wx[i][j]*lxg+self.wi[i][j]*lig+self.wf[i][j]*lfg+self.wo[i][j]*log
				self._wxg[i][j]+=lxg*xh[j]
				self._wig[i][j]+=lig*xh[j]
				self._wfg[i][j]+=lfg*xh[j]
				self._wog[i][j]+=log*xh[j]
		free(c)
		free(xh)
		free(ca)
		free(f)
		free(i_)
		free(o)
		free(out)
		free(self._hg)
		self._hg=hg
		return og



	def update(self,lr):
		self._sz=0
		free(self._cl)
		free(self._xhl)
		free(self._cal)
		free(self._fl)
		free(self._il)
		free(self._ol)
		free(self._outl)
		self._cl=None
		self._xhl=None
		self._cal=None
		self._fl=None
		self._il=None
		self._ol=None
		self._outl=None
		for i in range(0,self.y):
			self.bx[i]-=self._bxg[i]*lr
			self.bf[i]-=self._bfg[i]*lr
			self.bi[i]-=self._big[i]*lr
			self.bo[i]-=self._bog[i]*lr
			self._c[i]=0
			self._h[i]=0
			self._hg[i]=0
			self._cg[i]=0
			self._bxg[i]=0
			self._bfg[i]=0
			self._big[i]=0
			self._bog[i]=0
			for j in range(0,self.x+self.y):
				self.wx[i][j]-=self._wxg[i][j]*lr
				self.wf[i][j]-=self._wfg[i][j]*lr
				self.wi[i][j]-=self._wig[i][j]*lr
				self.wo[i][j]-=self._wog[i][j]*lr
				self._wxg[i][j]=0
				self._wfg[i][j]=0
				self._wig[i][j]=0
				self._wog[i][j]=0



	def save(self,f):
		a=array.array("f")
		a.fromlist(self.bx)
		f.write(a.tobytes())
		a=array.array("f")
		a.fromlist(self.bf)
		f.write(a.tobytes())
		a=array.array("f")
		a.fromlist(self.bi)
		f.write(a.tobytes())
		a=array.array("f")
		a.fromlist(self.bo)
		f.write(a.tobytes())
		for i in range(0,self.y):
			a=array.array("f")
			a.fromlist(self.wx[i])
			f.write(a.tobytes())
			a=array.array("f")
			a.fromlist(self.wf[i])
			f.write(a.tobytes())
			a=array.array("f")
			a.fromlist(self.wi[i])
			f.write(a.tobytes())
			a=array.array("f")
			a.fromlist(self.wo[i])
			f.write(a.tobytes())



	def reset_fd(self):
		for i in range(0,self.y):
			self._c[i]=0
			self._h[i]=0



class RNN:
	def __init__(self,fp,i,h,o,lr):
		self.fp=fp
		if (os.path.exists(self.fp)):
			with open(self.fp,"rb") as f:
				self.lstm=LSTMLayer(i,h,f)
				self.fc=FullyConnectedLayer(h,o,f)
		else:
			self.lstm=LSTMLayer(i,h,None)
			self.fc=FullyConnectedLayer(h,o,None)
		self.lr=lr



	def train(self,in_,t):
		l=malloc(len(in_))
		for i in range(0,len(in_)):
			l[i]=self.fc.train(self.lstm.forward(in_[i],tr=True),t[i],self.lr)
		for i in range(len(in_)-1,-1,-1):
			self.lstm.backward(l[i])
		self.lstm.update(self.lr)
		free(l)



	def predict(self,in_):
		for i in range(0,len(in_)-1):
			self.lstm.forward(in_[i])
		o=malloc(self.lstm.y)
		to=self.lstm.forward(in_[i])
		for i in range(0,self.lstm.y):
			o[i]=to[i]
		self.lstm.reset_fd()
		return self.fc.forward(o)



	def save(self):
		with open(self.fp,"wb") as f:
			self.lstm.save(f)
			self.fc.save(f)



N_SEQ=200
SEQ_LEN=20
N_EPOCH=20
DATA=[[[math.sin((i+j)*0.25)] for j in range(0,SEQ_LEN+1)] for i in range(0,N_SEQ)]
rnn=RNN("rnn-save.rnn",1,25,1,0.01)
for j in range(N_EPOCH):
	for i in range(N_SEQ):
		rnn.train(DATA[i][:SEQ_LEN],DATA[i][1:])
rnn.save()
