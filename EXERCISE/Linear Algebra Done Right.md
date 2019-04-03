## EXERCISES 1.A

------

**1**  Suppose $a​$ and $b​$ are real numbers, not both 0. Find real numbers $c​$ and $d​$ such that 
$$
1/(a+bi)=c+di
$$
**Solution:**
$$
\begin{equation}
\begin{aligned}
&\begin{aligned}
	\because\frac{1}{a+bi}&=\frac{a-bi}{(a+bi)(a-bi)}\\
	&=\frac{a-bi}{a^2+b^2}\\
	&=\frac{a}{a^2+b^2}+\frac{-b}{a^2+b^2}i\\
\end{aligned}\\
&\begin{aligned}
	\therefore c+di=\frac{a}{a^2+b^2}+\frac{-b}{a^2+b^2}i\\
\end{aligned}\\
&\begin{aligned}
	\therefore c=\frac{a}{a^2+b^2} \quad d=-\frac{b}{a^2+b^2}
\end{aligned}
\end{aligned}
\end{equation}
$$
**2** Show that
$$
\frac{-1+\sqrt{3}i}{2}
$$
is a cube root of 1 (meaning that its cube equals 1).

**Proof 1:**
$$
\begin{equation}
\begin{aligned}
\left(\frac{-1+\sqrt{3}i}{2}\right)^2&=\frac{1-3-2\sqrt{3}i}{4}=
\frac{-1-\sqrt{3}i}{2}\\
\Rightarrow\left(\frac{-1+\sqrt{3}i}{2} \right)^3&=\frac{-1-\sqrt{3}i}{2}\cdot\frac{-1+\sqrt{3}i}{2}\\
&=\frac{1+3}{4}\\
&=1
%\begin{aligned}
%\end{aligned}
\end{aligned}
\end{equation}
$$
**Proof 2:**
$$
\begin{equation}
\begin{aligned}
\frac{-1+\sqrt{3}i}{2}&=1\angle120^{\circ}\\
\left(\frac{-1+\sqrt{3}i}{2}\right)^3&=\left(1\angle120^{\circ}\right)^3\\
&=1\angle(120^{\circ}\cdot3)\\
&=1\angle360^{\circ}\\
&=1
\end{aligned}
\end{equation}
$$
**3** Find two distinct square roots of $\mathit{i}​$.

**Solution:**

Let $a+bi$ be the square root of $i$, where $a,b\in\mathbb{R}$, so that $\left(a+bi\right)^2=a^2-b^2+2abi=i$

Then, we can get equations about $a$ and $b$
$$
\left\{\begin{matrix}
\begin{aligned}
&a^2-b^2=0\\
&2ab=1
\end{aligned}
\end{matrix}\right.
$$
Solving equations and get
$$
\left\{\begin{matrix}
\begin{aligned}
&a=\frac{\sqrt{2}}{2}\\
&b=\frac{\sqrt{2}}{2}
\end{aligned}
\end{matrix}\right.
\quad
\left\{\begin{matrix}
\begin{aligned}
&a=-\frac{\sqrt{2}}{2}\\
&b=-\frac{\sqrt{2}}{2}
\end{aligned}
\end{matrix}\right.
$$
So the two distinct square roots of $i$ are $x_1=\frac{\sqrt{2}}{2}+\frac{\sqrt{2}}{2}$ and $x_2=-\frac{\sqrt{2}}{2}-\frac{\sqrt{2}}{2}​$

**4** Show that $\alpha+\beta=\beta+\alpha​$ for all $\alpha,\beta\in\mathbb{C}​$

**Proof:**

Let $\alpha=a+bi,\beta=c+di​$, where $a,b,c,d\in\mathbb{R}​$

So $\alpha+\beta=a+c+(b+d)i​$, $\beta+\alpha=c+a+(d+b)i​$

$\therefore \alpha+\beta=\beta+\alpha$, for all $\alpha,\beta\in\mathbb{C}$

**5** $\cdots$

**6** $\cdots$

**7** Show that for every $\alpha\in\mathbb{C}$, there exists a unique $\beta\in\mathbb{C}$ such that $\alpha+\beta=0$.

**Proof:**

Suppose that $\alpha$ is given, and $\alpha+\beta=0$, then let $\alpha=a+bi$, $\beta=c+di$.
$$
\alpha+\beta=0 \Rightarrow
\left\{\begin{matrix}
\begin{aligned}
&a+c=0\\
&b+d=0
\end{aligned}
\end{matrix}\right.
$$
There is one and only one set of solution for these equations
$$
\left\{\begin{matrix}
\begin{aligned}
&c=-a\\
&d=-b
\end{aligned}
\end{matrix}\right.
$$
So $\beta=-a-bi=-\alpha$ is the unique solution for $\alpha+\beta=0$.

**8** $\cdots$

**9** $\cdots$

**10** Find $x\in\mathbb{R}^4$ such that
$$
(4,-3,1,7)+2x=(5,9,-6,8).
$$
**Solution:**
$$
\begin{aligned}
(4,-3,1,7)+2x&=(5,9,-6,8)\\
2x&=(5,9,-6,8)-(4,-3,1,7)\\
2x&=(1,12,-7,1)\\
x&=(0.5,6,-3.5,0.5)
\end{aligned}
$$
**11** Explain why there does not exist $\lambda\in\mathbb{C}$ such that
$$
\lambda(2-3i,5+4i,-6+7i)=(12-5i,7+22i,-32-9i)
$$
**Proof:**

Suppose there exist $\lambda=a+bi\in\mathbb{C}$, which satisfies the equation above.

So we can get the equations about $\lambda​$
$$
\left\{\begin{matrix}
\begin{aligned}
&(a+bi)\cdot(2-3i)=12-5i\\
&(a+bi)\cdot(5+4i)=7+22i\\
&(a+bi)\cdot(-6+7i)=-32-9i
\end{aligned}
\end{matrix}\right.
$$
Obviously, the equations set above does not have a solution, so that such $\lambda$ does not exist.

**12** $\cdots$

**13** $\cdots$

**14** $\cdots$

**15** $\cdots$

**16** $\cdots​$

## EXERCISES 1.B

------

**1** Prove that $-(-v)=v$ for every $v\in\mathbb{V}$.

**Proof:**
$$
\begin{aligned}
-(-v)&=(-1)\cdot(-1)v\\
&=((-1)\cdot(-1))v\\
&=v
\end{aligned}
$$
**2** $\cdots$

**3** $\cdots$

**4** The empty set is not a vector space. The empty set fails to satisfy only one of the requirements listed in 1.19. Which one?

**Solution:** It fails to satisfy addictive identity, becuase $0\notin\varnothing$.

**5** $\cdots$

**6** $\cdots$

## EXERCISES 1.C

------

**1** For each of the following subsets of $\mathbf{F}^3$, determine whether it is a subspace of $\mathbf{F}^3$:

(a)	$\{(x_1,x_2,x_3)\in\mathbf{F}^3:x_1+2x_2+3x_3=0\};$

**Solution:** 

This is a subspace of $\mathbf{F}^3​$.

Because $(0,0,0)\in(a)$, $u,v\in(a)$, then $u+v\in(a)$.

$\cdots​$

**2** $\cdots$

**3** Show that the set of differentiable real-value functions $f$ on the interval $(-4,4)$ such that $f'(-1)=3f(2)$ is a subspace of $\mathbf{R}^{(-4,4)}$.

**Proof**

Denote the target subspace is $V$.

Obviously, $f\equiv 0\in V​$

If $f,g\in V$, then $(f+g)'(-1)=f'(-1)+g'(-1)=3f(2)+3g(2)=3(f+g)(2)$, $f+g\in V$.

If $\lambda\in\mathbf{R}$, then $(\lambda f)'(-1)=\lambda\cdot f'(-1)=\lambda\cdot3f(2)=3(\lambda f)(2)$, $\lambda f\in V$

