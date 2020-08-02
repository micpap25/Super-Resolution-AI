import aiohttp
import uvicorn
from fastai.callbacks import *
from fastai.vision import *
from starlette.applications import Starlette
from starlette.responses import Response, HTMLResponse, RedirectResponse
from torchvision.models import vgg16_bn
from FeatureLoss import FLRunner


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()


@app.route("/upload", methods=["POST"])
async def upload(request):
    formdata = await request.form()
    filename = formdata["file"].filename
    imgbytes = await (formdata["file"].read())
    return predict_image_from_bytes(imgbytes, filename)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(imgbytes, filename=""):
    path = Path('data/images')
    img = PIL.Image.open(BytesIO(imgbytes))
    if len(filename) > 0:
        img.filename = filename
    else:
        img.filename = 'url_picture.jpeg'
    img.save(path / img.filename)
    src = ImageImageList.from_folder(path).split_by_rand_pct(0.1, seed=42)
    imgdata = (src.label_from_func(lambda x: path / x.name).transform(get_transforms(), size=(img.size[1], img.size[0]),
                                                                      tfm_y=True).databunch(bs=1).normalize(
        imagenet_stats, do_y=True))
    arch = models.resnet34
    wd = 1e-3
    vgg_m = vgg16_bn(True).features.eval()
    blocks = [i - 1 for i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)]
    requires_grad(vgg_m, False)
    feat_loss = partial(FLRunner, vgg_m, blocks[2:5], [5, 15, 2])
    learn = unet_learner(imgdata, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics, blur=True,
                         norm_type=NormType.Weight)
    learn.path = Path('data')
    learn.load('rezolute256')
    fastimg = open_image(BytesIO(imgbytes))
    fastimg.save(path / 'fastimg.jpeg')
    pred = learn.predict(fastimg)[0]
    pred.save(path / 'pred.jpeg')
    img_bytes = BytesIO()
    pred.save_with_format(img_bytes, format='jpeg')
    # comment these lines out if you want to save the image
    for f in os.listdir(path):
        os.remove(path / f)
    learn.destroy()

    return Response(img_bytes.getvalue(), media_type='image/jpeg')


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
