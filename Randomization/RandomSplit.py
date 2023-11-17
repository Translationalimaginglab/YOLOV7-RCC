# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# January 2023
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import numpy as np

def RandomSplit(W, training_size, tries=10):
    assert W.ndim == 2
    
    N = W.shape[1]
    
    if training_size < 1:
        training_size = int(training_size*N)
        
    assert training_size > 0 and training_size <= N
    
    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something
    
    if training_size == N:
        return np.ones(N, dtype=int)
    
    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    assert K > 1 and N >= K
    
    D = 1.0/D
    Z = np.eye(K) - 1.0/K
    
    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]
    
    # This is ZDW
    W = Z @ W

    U, S, Vh = np.linalg.svd(W)
    
    Q = Vh[(K-1):, :].T
    
    bestRes = -1.0
    bestX = None
    
    for _ in range(tries):
        x = np.random.randn(Q.shape[1])
        x = np.inner(Q, x)
        
        ind = np.argsort(x)
        ind = ind[::-1]
        x = np.zeros(N, dtype=int)
        x[ind[:training_size]] = 1
       
        res = np.linalg.norm(np.inner(W, x))
        
        if res < bestRes or bestRes < 0.0:
            bestRes = res
            bestX = x
    
    return bestX, bestRes
