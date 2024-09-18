1. Let $\mathbf{q} = \begin{bmatrix}x_b \\ y_b \\ z_b \\ \theta_b \\
   \psi_1 \\ \psi_2 \\ \psi_3 \\ \psi_4 \\ \psi_5 \\ \psi_6 \\ \psi_7 \end{bmatrix}$
   be the set of joint values that places the end effector at some desired position.
   Notice that for a given base position $\mathbf{x_b} = [x_b, y_b, z_b, \theta_b]^T$, the set of (arm) joint values that places the end effector at a desired position
   is characterized by the elbow joint tracing out a circle.  Parameterize this circle with $n$.
   We then have
   $\mathbf{q} = \mathbf{f}(n, x_b, y_b, z_b, \theta_b)$ for $n \in [0, 1]$

   Equation of a unit circle lying on the y-z plane (pointing along +x axis): $\bar c(n) = \begin{bmatrix}
     0 \\
     \cos(n) \\
     \sin(n)
   \end{bmatrix}$

   $\mathbf{p}(n, x_b, y_b, z_b, x_e, y_e, z_e)$ defines a point
   on a circle $\mathbf{c}(d, \phi)$ where $d$ is the distance between $\mathbf{x_b} = (x_b, y_b, z_b)$ and $\mathbf{x_e} = (x_e, y_e, z_e)$
   and $\phi$ is the defined by the direction pointing from $\mathbf{x_b}$ to $\mathbf{x_e}$.
   $ \mathbf{c}(d, \phi) = R(\phi) \cdot T(\frac{d}{2}, 0, 0) \cdot S(r(d)) \bar c$

   - Let $\Delta = \mathbf{x_e} - \mathbf{x_b}$
   - $d = \lVert\Delta\rVert _2$
   - $\phi = \frac{\Delta}{\lVert\Delta\rVert _2}$
   - $R(\phi)$ is the rotation matrix that rotates the vector $[1, 0, 0]^T$ to $\phi$, (**IS THIS UNIQUE & SMOOTH?!**)
     (https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d)
     - $v = \begin{bmatrix}0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0\end{bmatrix} \phi = \begin{bmatrix} 0 \\ -\phi_3 \\ \phi_2\end{bmatrix}$
     - $[v]_\times = \begin{bmatrix}
       0 & -\phi_2 & -\phi_3 \\
       \phi_2 & 0 & 0 \\
       \phi_3 & 0 & 0 \\
     \end{bmatrix}$
     - $[v]_\times^2 = \begin{bmatrix}
       -\phi_2^2 - \phi_3^2 & 0 & 0 \\
       0 & -\phi_2^2 & -\phi_2 \phi_3 \\
       0 & -\phi_2 \phi_3 & -\phi_3^2 \\
     \end{bmatrix}$
     - $c = \phi_1$
     - $ R(\phi) = I + [v]_\times + [v]_\times^2 \frac{1}{1+c}
       = \begin{bmatrix}
       1 - \frac{\phi_2^2 + \phi_3^2}{1+\phi_1} & -\phi_2 & -\phi_3 \\
       \phi_2 & 1 - \frac{\phi_2^2}{1+\phi_1} & -\frac{\phi_2\phi_3}{1+\phi_1} \\
       \phi_3 & -\frac{\phi_2\phi_3}{1+\phi_1} & 1 - \frac{\phi_3^2}{1+\phi_1}
     \end{bmatrix}$
   - $T(x,y,z)$ is the translation in $x, y, z$ direction
     - $T(x,y,z) = \begin{bmatrix}1 & 0 & 0 & x \\ 0 & 1 & 0 & y \\ 0 & 0 & 1 & z \\ 0 & 0 & 0 & 1\end{bmatrix}$
   - $S(r)$ scales uniformly by $r$ and $r(d)$ is the solution to the equation
     $d = \sqrt{l_1^2 - r^2} + \sqrt{l_2^2 - r^2}$
     $$
     d^2 = (l_1^2 - r^2) + (l_2^2 - r^2) + 2\sqrt{(l_1^2 - r^2)(l_2^2 - r^2)}
     $$
     (For $l_1 = l_2 = 1$, $r = \sqrt{1-\frac{d^2}{4}}$)
     - $S(r) = \begin{bmatrix}r & 0 & 0 & 0 \\ 0 & r & 0 & 0  \\ 0 & 0 & r & 0 \\ 0 & 0 & 0 & 1\end{bmatrix}$
   - Let $l_{1\perp} = \sqrt{l_1^2 - r^2}$
   - $\mathbf{c}(x_b, x_e) = T(x_b)R(\phi) S(r) T(l_{1\perp}, 0, 0) \begin{bmatrix}
       0 \\ \cos(n) \\ \sin(n) \\ 1
     \end{bmatrix} =
     \left[\begin{matrix}l_{1\perp} \left(1 + \frac{- \phi_{2}^{2} - \phi_{3}^{2}}{\phi_{1} + 1}\right) - \phi_{2} r \cos{\left(n \right)} - \phi_{3} r \sin{\left(n \right)} + x_{b1}\\l_{1\perp} \phi_{2} - \frac{\phi_{2} \phi_{3} r \sin{\left(n \right)}}{\phi_{1} + 1} + r \left(- \frac{\phi_{2}^{2}}{\phi_{1} + 1} + 1\right) \cos{\left(n \right)} + x_{b2}\\l_{1\perp} \phi_{3} - \frac{\phi_{2} \phi_{3} r \cos{\left(n \right)}}{\phi_{1} + 1} + r \left(- \frac{\phi_{3}^{2}}{\phi_{1} + 1} + 1\right) \sin{\left(n \right)} + x_{b3}\\1\end{matrix}\right]
     $

     (see `eq1`) (Note can be further simpified with https://en.wikibooks.org/wiki/Trigonometry/Simplifying_a_sin(x)_%2B_b_cos(x))
     - $0 < d < l_1 + l_2$
