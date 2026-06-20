# Enfoque de solución
Para el uso más optimizado y eficiente de las herramientas de SAM3, YOLO y ByteTrack se tomo en cuenta una arquitectura que aplicara el menor esfuerzo
del hardware en el uso de SAM3, ya que se busca la replicación de esta misma solución con computo accesible para alguien con una computadora y GPU
de NVIDIA o el uso de herramientas en la nube sin que resulte tan exigente la aplicación de estas mismas.

## Arquitectura
En este código se usó el siguiente pipeline para el trackeo de robots y balón:
YOLO -> ByteTrack -> SAM3
Esto con la idea de la carga de trabajo de SAM3 y evitar en mayor medida la autonomía no regularizada del mismo, así como excluir el uso de herramientas
del algoritmo como su códificación a través de prompts de texto, ya que estos repercuten exponencialmente en el poder de procesamiento y como consecuencia
se perderia la visión elevando el costo de la solución

# Instalación
## Descargar el modelo SAM3
Como paso inicial ya que se necesita tanto para entorno local como virtual descargaremos el modelo de SAM3 desde el repositorio en huggingface, pero para esto
se tiene que pedir primero permiso para usar el mismo en el siguiente enlace: https://huggingface.co/facebook/sam3
una vez que se acrediten los permisos, dentro de "Files and versions" en el repositorio descargamos el archivo sam3.pt que contiene el modelo predeterminado de SAM3

## Creación de un Acces Token
Dentro de nuestra cuenta de huggingface en la sección de Acces Token tenemos que crear uno nuevo donde eligamos su nombre y darle los permisos para acceder a nuestsros repositorios
si se tienen más de uno y solo se desea usar el de SAM3 podemos darle permiso solo a ese

## Para el caso local
### Creación de un entorno virtual
Con la intención de evitar cualquier problema con incompatibilidad en versiones de librerias y programas del sistema con las que se usaron para la creación de esta solución
creamos un entorno virtual con el uso de miniconda que se puede obtener en este enlace: https://www.anaconda.com/download

Después creamos el entorno dentro de nuestra terminal para luego activarlo e instalar las librerías correspondientes teniendo en cuenta que se tiene una GPU NVIDIA
```bash
conda create -n supervision python=3.11
```

```bash
conda activate supervision
```
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install supervision ultralytics trackers
pip install jupyter
```
Luego, usamos el comando
```bash
hf auth login
```
para después pegar con solo click derecho el token que copiamos anteriormente cuando creamos este en nuestra cuenta de huggingface
### Usar entorno de programación
Ahora vamos a pasar a nuestra aplicación para codificación donde vamos a utilizar python, en este caso ya sea VS Code o Visual Studio podemos utilizarlas teniendo en cuenta que
el propio python tiene que estar instalado asi como creado nuestro entorno virtual para poder seleccionar en nuestro kernel en la esquina superior derecha  **Select Kernel** → **Python Environments** → `supervision`.
donde si es la primera vez que lo hacemos nos pedira instalar automáticamente las dependencias que va a usar

## Para el caso virtual
Para esta opción vamos a usar Google Colab, ya que es la que nosotros utilizamos durante las pruebas de nuestro código, primero vamos a la página: https://colab.research.google.com/
siguiente crearemos un nuevo nueo notebook donde cambiaremos nuestro entorno de ejecución (Runtime) a uno que tenga una GPU para poder usar al máximo SAM3, si tenemos beneficios premium de colab 
podemos seleccionar una GPU más potente, pero en defecto usamos una T4 GPU. Para que podamos acceder a SAM3 desde colab la manera más conveniente es subir el archivo SAM3.pt a una carpeta de drive 
junto con los archivos que se desean utilizar y montarla cada notebook para que cada sesión de colab no se tenga que subir los videos o que descarge el modelo de sam3.pt que pesa ~3.4 GB. 
Y como último paso antes de usar el código se ejecuta lo siguiente

```bash
!pip install opencv-python transformers torch torchvision ultralytics supervision trackers
```
para luego abrir la terminal de colab y escribir

```bash
hf auth login
```
y pegar nuestro token de nuestra cuenta de huggingface.

### Reel de Instagram
https://www.instagram.com/reel/DZzAPXSAKBG/?igsh=MXBpdHF6eDc4enN1cA==
