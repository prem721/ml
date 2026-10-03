# Week 7: Exp1 AND gate, Exp2 OR gate, Exp3 MLP backpropagation (5 epochs)
import numpy as np


def step(z):
    return 1 if z >= 0 else 0


def gate(name, w, b):
    print(f"--- {name} gate (w={w}, b={b}) ---")
    print("x1 x2 |   z   | out")
    for x1 in (0, 1):
        for x2 in (0, 1):
            z = w[0] * x1 + w[1] * x2 + b
            print(f" {x1}  {x2} | {z:5.1f} |  {step(z)}")
    print()


gate("AND", (1, 1), -1.5)   # Exp 1
gate("OR", (1, 1), -0.5)    # Exp 2

# ---------- Exp 3: MLP, sigmoid, target=0.5, lr=1, no biases (as in figure) ----------
sig = lambda z: 1 / (1 + np.exp(-z))

x = np.array([0.35, 0.7])
W1 = np.array([[0.2, 0.3],    # x1 -> h1 (w11), x1 -> h2 (w12)
               [0.2, 0.3]])   # x2 -> h1 (w21), x2 -> h2 (w22)
W2 = np.array([0.3, 0.9])     # h1 -> o (w13), h2 -> o (w23)
target, lr = 0.5, 1.0

for epoch in range(1, 6):
    # forward
    h = sig(x @ W1)
    out = sig(h @ W2)
    loss = 0.5 * (target - out) ** 2
    # backward
    d_out = (out - target) * out * (1 - out)
    d_h = d_out * W2 * h * (1 - h)
    W2 = W2 - lr * d_out * h
    W1 = W1 - lr * np.outer(x, d_h)
    print(f"Epoch {epoch}: Actual={out:.4f} Target={target} Loss={loss:.6f}")
    print(f"   w11={W1[0,0]:.4f} w12={W1[0,1]:.4f} w21={W1[1,0]:.4f} w22={W1[1,1]:.4f}"
          f" w13={W2[0]:.4f} w23={W2[1]:.4f}")
