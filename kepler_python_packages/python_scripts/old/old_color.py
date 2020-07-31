def color_convert(
    *args,
    **kwargs):
    """
    convert between color systems

    allowed are ic/oc = 
    0: RGB
    1: HSV 
    2: HSL
    3: YCH
    4: CMY
    5: YIQ
    6: YUV
    7: YUV_
    8: YCbCr
    9: YPbPr
    """
    fm = {'RGB'  :   rgb2rgb,
          'HSV'  :   hsv2rgb,
          'HSL'  :   hsl2rgb,
          'YCH'  :   ych2rgb,
          'CMY'  :   cmy2rgb,
          'YIQ'  :   yiq2rgb,
          'YUV'  :   yuv2rgb,
          'YUV_' :  yuv_2rgb,
          'YCbCr': YCbCr2rgb,
          'YPbPr': YPbPr2rgb}
    tm = {'RGB'  : rgb2rgb,
          'HSV'  : rgb2hsv,
          'HSL'  : rgb2hsl,
          'YCH'  : rgb2ych,
          'CMY'  : rgb2cmy,
          'YIQ'  : rgb2yiq,
          'YUV'  : rgb2yuv,
          'YUV_' : rgb2yuv_,
          'YCbCr': rgb2YCbCr,
          'YPbPr': rgb2YPbPr}
    # translate to RGB in between.
    ic = kwargs.get('ic','RGB')
    oc = kwargs.get('oc','RGB')     
    ci = args[0:3]
    cm = fm[ic](*ci)
    co = tm[oc](*cm)
    return co

def rgb2rgb(*args):
    """
    dummy
    """
    return args

def hsv2rgb(h,s,v):
    """
    hue = [0, 360], saturation = [0, 1], value =[0, 1]
    """
    h = np.mod(h, 360.)
    s = min(max(s,0),1)
    v = min(max(v,0),1)
    c = v * s
    p = h / 60.
    x = c * (1 - np.abs(np.mod(p, 2.) - 1.))
    if p < 1:
        r,g,b = c,x,0
    elif p < 2:
        r,g,b = x,c,0
    elif p < 3:
        r,g,b = 0,c,x
    elif p < 4:
        r,g,b = 0,x,c
    elif p < 5:
        r,g,b = x,0,c
    else:
        r,g,b = c,0,x
    m = v - c
    r,g,b = r+m,g+m,b+m
    return r,g,b

def rgb2hsv(r,g,b):
    """
    hue = [0, 360], saturation = [0, 1], value =[0, 1]
    """
    M = max(r,g,b)
    m = min(r,g,b)
    C = M - m
    h = 0
    if M == r:
        h = (g-b)/C % 6
    elif M == g:
        h = (b-r)/C + 2 
    elif M == b:
        h = (b-r)/C + 4
    H = h * 60
    V = M
    S = 0 if C == 0 else C/V
    return H,S,V


def hsl2rgb(h,s,l):
    """
    hue = [0, 360], saturation = [0, 1], lightness =[0, 1]
    """
    h = np.mod(h, 360.)
    s = min(max(s,0),1)
    l = min(max(l,0),1)
    c = (1 - abs(2 * l - 1)) * s
    p = h / 60.
    x = c * (1 - abs(np.mod(p, 2.) - 1.))
    if p < 1:
        r,g,b = c,x,0
    elif p < 2:
        r,g,b = x,c,0
    elif p < 3:
        r,g,b = 0,c,x
    elif p < 4:
        r,g,b = 0,x,c
    elif p < 5:
        r,g,b = x,0,c
    else:
        r,g,b = c,0,x
    m = l - 0.5 * c
    r,g,b = r+m,g+m,b+m
    return r,g,b

def rgb2hsl(r,g,b):
    """
    hue = [0, 360], saturation = [0, 1], lightness =[0, 1]
    """
    M = max(r,g,b)
    m = min(r,g,b)
    C = M - m
    h = 0
    if M == r:
        h = (g-b)/C % 6
    elif M == g:
        h = (b-r)/C + 2 
    elif M == b:
        h = (b-r)/C + 4
    H = h * 60
    L = 0.5*(M + m)
    S = 0 if C == 0 else C/(1-np.abs(2*L-1))
    return H,S,L

def hsi2rgb(h,s,i):
    """
    hue = [0, 360], saturation = [0, 1], intensity =[0, 1]
    """
    h = np.mod(h, 360.)
    s = min(max(s,0),1)
    i = min(max(i,0),1)

    p = h / 60.
    f = 0.5 * np.mod(p, 2.)
    c = s * i * 3
    x = c * (1 - f)
    y = c * f

    if p < 2:
        r,g,b = x,y,0
    elif p < 4:
        r,g,b = 0,x,y
    else:
        r,g,b = y,0,x
    m = i - i * s
    r,g,b = r+m,g+m,b+m
    r = min(max(r,0),1)
    g = min(max(g,0),1)
    b = min(max(b,0),1)
    return r,g,b


def rgb2hsi(r,g,b):
    """
    hue = [0, 360], saturation = [0, 1], intensity =[0, 1]
    """
    r = min(max(r,0),1)
    g = min(max(g,0),1)
    b = min(max(b,0),1)
    M = max(r,g,b)
    m = min(r,g,b)
    C = M - m
    h = 0
    if M == r:
        h = (g-b)/C % 6
    elif M == g:
        h = (b-r)/C + 2 
    elif M == b:
        h = (b-r)/C + 4
    H = h * 60
    I = (r + g + b) / 3.
    S = 0 if C == 0 else 1 - m/I
    return H,S,I


def ych2rgb(y,c,h):
    """
    hue = [0, 360], chroma = [0, 1], luma =[0, 1]
    """
    h = np.mod(h, 360.)
    c = min(max(c,0),1)
    y = min(max(y,0),1)
    p = h / 60.
    x = c * (1 - abs(np.mod(p, 2.) - 1.))
    if p < 1:
        r,g,b = c,x,0
    elif p < 2:
        r,g,b = x,c,0
    elif p < 3:
        r,g,b = 0,c,x
    elif p < 4:
        r,g,b = 0,x,c
    elif p < 5:
        r,g,b = x,0,c
    else:
        r,g,b = c,0,x
    m = y - (0.30*r + 0.69*g + 0.11*b)
    r,g,b = r+m,g+m,b+m
    return r,g,b

def rgb2ych(r,g,b):
    """
    hue = [0, 360], chroma = [0, 1], luma =[0, 1]
    """
    M = max(r,g,b)
    m = min(r,g,b)
    C = M - m
    h = 0
    if M == r:
        h = (g-b)/C % 6
    elif M == g:
        h = (b-r)/C + 2 
    elif M == b:
        h = (b-r)/C + 4
    H = h * 60
    y = (0.30*r + 0.69*g + 0.11*b)
    return y,C,H

def cmy2rgb(c,m,y):
    """
    c,m,y = [0, 1]
    """
    c = min(max(c,0),1)
    l = min(max(l,0),1)
    y = min(max(y,0),1)
    r,g,b = 1-c,1-m,1-y
    return r,g,b
         
def rgb2cmy(r,g,b):
    """
    c,m,y = [0, 1]
    """
    r = min(max(r,0),1)
    g = min(max(g,0),1)
    b = min(max(b,0),1)
    c,m,y = 1-r,1-g,1-b
    return r,g,b
         
def yiq2rgb(y,i,q):
    """
    y = [0, 1]
    |i| <= 0.596
    |q| <= 0.523
    
    TODO - replace by explicit matrix
    """
    a = np.matrix([[0.299, 0.587, 0.114],
                   [0.596,-0.275,-0.321],
                   [0.212,-0.523, 0.311]]).getI()
    r,g,b = np.inner(a,np.array([y,i,q])).tolist()
    return r,g,b

def rgb2yiq(r,g,b):
    """
    y = [0, 1]
    |i| <= 0.596
    |q| <= 0.523
    """
    a = np.matrix([[0.299, 0.587, 0.114],
                   [0.596,-0.275,-0.321],
                   [0.212,-0.523, 0.311]])
    r,g,b = np.inner(a,np.array([r,g,b])).tolist()
    return y,i,q

def yuv2rgb(y,u,v):
    """
    y = [0 ,1]
    |u| <= 0.436
    |v| <= 0.615

    Rec. 601
    """
    a = np.matrix([[ 0.299  , 0.587   , 0.114  ],
                   [-0.14713,-0.28886 , 0.463  ],
                   [ 0.615  ,-0.551499,-0.10001]])
    r,g,b = np.inner(a,np.array([y,u,v])).tolist()
    return r,g,b

def rgb2yuv(y,u,v):
    """
    y = [0 ,1]
    |u| <= 0.436
    |v| <= 0.615

    Rec. 601
    """
    a = np.matrix([[ 1, 0      , 1.13983],
                   [ 1,-0.39465,-0.58060],
                   [ 1, 2.03211, 0      ]])
    y,u,v = np.inner(a,np.array([r,g,b])).tolist()
    return y,u,v

def yuv_2rgb(y,u,v):
    """
    y = [0 ,1]
    |u| <= 0.436?
    |v| <= 0.615?

    Rec. 709
    """
    a = np.matrix([[ 0.2126 , 0.7152  , 0.0722 ],
                   [-0.09991,-0.33609 , 0.436  ],
                   [ 0.615  ,-0.55861 ,-0.05639]])
    r,g,b = np.inner(a,np.array([y,u,v])).tolist()
    return r,g,b

def rgb2yuv_(r,g,b):
    """
    y = [0 ,1]
    |u| <= 0.436?
    |v| <= 0.615?

    Rec. 709
    """
    a = np.matrix([[ 1, 0      , 1.28033],
                   [ 1,-0.21482,-0.38059],
                   [ 1, 2.12798, 0      ]])
    y,u,v = np.inner(a,np.array([r,g,b])).tolist()
    return y,u,v

def rgb2YCbCr(r,g,b):
    """
    y = floating-point value between 16 and 235
    Cb,Cr floating-point values between 16 and 240
    """
    a = np.matrix([[ 0.299  , 0.587   , 0.114  ],
                   [-0.169  ,-0.331   , 0.499  ],
                   [ 0.499  ,-0.418   ,-0.0813 ]])
    Y,Cb,Cr = (np.inner(a,np.array([r,g,b]))*256 + 
               np.array([0.,128.,128.])).tolist()
    return Y,Cb,Cr

def YCbCr2rgb(Y,Cb,Cr):
    """
    y = floating-point value between 16 and 235
    Cr,Cb floating-point values between 16 and 240
    """
    a = np.matrix([[ 1, 0    , 1.402  ],
                   [ 1,-0.344,-0.714  ],
                   [ 1,+1.772, 0      ]])
    r,g,b = (np.inner(a,np.array([Y,Cb,Cr]) - np.array([0.,128.,128.]))
               / 256.).tolist()
    return r,g,b

def rgb2YPbPr(r,g,b):
    """
    y = [0, 1]
    Pb,Pr = [-0.5, 0.5]
    """
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Pb = b - Y
    Pr = r - Y
    return Y,Pb,Pr


def YPbPr2rgb(Y,Pb,Pr):
    """
    y = [0, 1]
    Pb,Pr = [-0.5, 0.5]
    """
    R = 0.2126 
    G = 0.7152 
    B = 0.0722
    a = np.matrix([[  R,  G,  B],
                   [ -R, -G,1-B],
                   [1-R, -G, -B]]).getI()
    r,g,b = np.inner(a, np.array([Y,Pb,Pr])).tolist()
    return r,g,b
