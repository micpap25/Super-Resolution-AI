import aiohttp
import uvicorn
from fastai.vision import *
from starlette.applications import Starlette
from starlette.responses import Response, HTMLResponse, RedirectResponse

from fastai.callbacks import *
from fastai.utils.mem import *


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


base_loss = F.l1_loss


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


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
    # TODO: find the image's extension and use it everywhere
    # TODO: test if .pkl works better
    # TODO: figure out what the windowsPath error is (different python/pytorch versions? path issue?)
    if len(filename) > 0:
        img.filename = filename
    else:
        img.filename = 'url_picture.jpeg'
    img.save(path / img.filename)
    src = ImageImageList.from_folder(path).split_none()
    imgdata = (src.label_from_func(lambda x: path / x.name).transform(get_transforms(), size=(
        img.size[1], img.size[0]), tfm_y=True).databunch(bs=1).normalize(imagenet_stats, do_y=True))
    learn = load_learner("data", "rezolute256.pkl")
    learn.data = imgdata
    fastimg = open_image(BytesIO(imgbytes))
    fastimg.save(path / 'fastimg.jpeg')
    pred = learn.predict(fastimg)[0]
    pred.save(path / 'pred.jpeg')
    img_bytes = BytesIO()
    pred.save_with_format(img_bytes, format='jpeg')
    # comment these lines out if you want to save the image
    # for f in os.listdir(path):
    #    os.remove(path / f)
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
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
