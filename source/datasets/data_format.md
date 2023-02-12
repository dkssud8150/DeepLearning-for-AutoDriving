# Image to tensor

Image 메소드에서 tensor로 바로 변경은 불가능하여 numpy로 변경 후 tensor로 바꿔줘야 한다.

```python
import torch
import numpy as np
from PIL import Image

src = Image.new(mode="RGB", size=(20,20))
src = np.asarray(src)
tns = torch.tensor(src, dtype=torch.uint8)
```

&nbsp;

&nbsp;

# torch.zeros()

tensor 형태로 만들 떄, torch.zero(5) 를 사용할 때가 있고, torch.zeros((5, ))를 사용할 때가 있다. 이 둘의 차이점을 확인해보자.

```python
import torch

n = 5

a = torch.zeros((n, ))
b = torch.zeros(n)

print(a,b,c)
print(a.shape, b.shape, c.shape)

# ========================================
# tensor([0., 0., 0., 0., 0.]) tensor([0., 0., 0., 0., 0.]) 
# torch.Size([5]) torch.Size([5])
```

결론적으로, a와 b는 같은 형태와 같은 사이즈를 가지고 있다.

&nbsp;

&nbsp;

# Tensor & tensor & as_tensor

참고 블로그 : [https://velog.io/@minchoul2/torch.Tensor%EC%99%80-torch.tensor%EC%9D%98-%EC%B0%A8%EC%9D%B4](https://velog.io/@minchoul2/torch.Tensor%EC%99%80-torch.tensor%EC%9D%98-%EC%B0%A8%EC%9D%B4)

&nbsp;

tensor 형태로 만드는 방법에는 여러 가지가 있다. 따라서 이 메소드들의 차이점을 확인해보고자 한다.

```python
import torch
import numpy as np

n = 5

l = [0,0,0,0,0]
nu = np.zeros((n,))
a = torch.zeros((n, ))


d = torch.Tensor(nu) # Class
e = torch.tensor(nu) # Function
f = torch.as_tensor(nu) # Copy=False, tensor의 dtype이나 device를 변경하기 위해서 사용 == .to(), .cpu(), .cuda(), .float(), .double()와 동일, copy=True로 하려면 torch.new_tensor()

nu[0] = 1

print(d,e,f)

'''
input : list
- Tensor : Copy=True
- tensor : Copy=True
- as_tensor : Copy=True

result : tensor([0., 0., 0., 0., 0.]) tensor([0, 0, 0, 0, 0]) tensor([0, 0, 0, 0, 0])


input : numpy
- Tensor : Copy=True
- tensor : Copy=True
- as_tensor : Copy=False

result : tensor([0., 0., 0., 0., 0.]) tensor([0., 0., 0., 0., 0.], dtype=torch.float64) tensor([1., 0., 0., 0., 0.], dtype=torch.float64)

input : tensor
- Tensor : Copy=False
- tensor : Copy=True
- as_tensor : Copy=False

result : tensor([1., 0., 0., 0., 0.]) tensor([0., 0., 0., 0., 0.]) tensor([1., 0., 0., 0., 0.])
'''
```

일단 Tensor와 tensor는 Class냐 Function이냐의 차이가 있고, input data의 포맷에 따라 copy를 할지말지에 대한 차이가 있다. Copy=True 일 경우에는 새로운 메모리 공간을 할당하여 새로운 배열을 생성한다. 반대로 Copy=False일 경우에는 기존의 input data 메모리 공간을 그대로 사용한다. 따라서 input data의 값을 변경하면 True인 데이터는 변하지 않고, False인 데이터는 함께 바뀐다.

- `tensor`의 경우는 항상 Copy=True
- input이 list일 경우에는 다 Copy=True로 설정된다.
- input이 numpy array일 경우에는 `as_tensor`만 Copy=False이고, 나머지는 새로운 공간을 할당한다.
- input이 tensor일 경우에는 `tensor`만 Copy=True 이고, 나머지는 기존의 메모리 공간을 사용한다.
- input이 tensor일 경우 저런 방법말고, 간단하게 `.clone()` 을 사용하면 된다.

&nbsp;

추가적으로 `as_tensor`는 원래 tensor의 dtype이나 device를 변경할 때 주로 사용한다. 이는 `.to()`, `.cpu()`, `.cuda()`, `.float()` 등과 동일하다. copy=True로 하고자 할 때 torch.new_tensor() 를 사용할 수도 있다.

`torch.from_numpy()` 를 통해 numpy를 tensor로 바꿔줄수도 있다. 메모리는 원래 메모리를 상속받는다.

&nbsp;

