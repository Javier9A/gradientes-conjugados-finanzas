# gradientes-conjugados-finanzas
# Aplicación del Método de Gradientes Conjugados Precondicionado en la Planificación Financiera

Desafío de Métodos Numéricos · UMSA · Carrera de Informática · Mención Ingeniería de Sistemas
Estudiante: Javier Averanga Quelka

## Descripción
Página web interactiva que resuelve Ax = b para distribuir presupuestos (en miles de Bs)
entre infraestructura, desarrollo de software, seguridad informática y soporte técnico.
Compara Cholesky (referencia), Gradientes Conjugados (GC) y GC Precondicionado con Jacobi
en tres escenarios: ideal, bajo estrés financiero y mal condicionado.

## Arquitectura
- `index.html`: estructura de las 10 secciones.
- `style.css`: diseño responsive.
- `script.js`: núcleo numérico (Cholesky, GC, GCP, autovalores, validación SPD),
  interfaz, gráfico de convergencia en canvas y módulo QA.

## Ejecución local
No requiere instalación. Descargar los tres archivos en una misma carpeta
y abrir `index.html` en el navegador.

## Enlace público
(pegar aquí la URL de GitHub Pages)

## Referencias
- Suñagua S., P. (2020). Método de Gradientes Conjugados Precondicionado. Revista Boliviana de Matemática – UMSA, 04, 2–7.
- Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM.
