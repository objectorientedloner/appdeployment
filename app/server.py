from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://www.dropbox.com/s/w096x0k6yt6yaqj/trial-slicing-2.pth?dl=0'
model_file_name = 'model'
classes = ['Abra',
 'Aerodactyl',
 'Alakazam',
 'Alolan Sandslash',
 'Arbok',
 'Arcanine',
 'Articuno',
 'Beedrill',
 'Bellsprout',
 'Blastoise',
 'Bulbasaur',
 'Butterfree',
 'Caterpie',
 'Chansey',
 'Charizard',
 'Charmander',
 'Charmeleon',
 'Clefable',
 'Clefairy',
 'Cloyster',
 'Cubone',
 'Dewgong',
 'Diglett',
 'Ditto',
 'Dodrio',
 'Doduo',
 'Dragonair',
 'Dragonite',
 'Dratini',
 'Drowzee',
 'Dugtrio',
 'Eevee',
 'Ekans',
 'Electabuzz',
 'Electrode',
 'Exeggcute',
 'Exeggutor',
 'Farfetchd',
 'Fearow',
 'Flareon',
 'Gastly',
 'Gengar',
 'Geodude',
 'Gloom',
 'Golbat',
 'Goldeen',
 'Golduck',
 'Golem',
 'Graveler',
 'Grimer',
 'Growlithe',
 'Gyarados',
 'Haunter',
 'Hitmonchan',
 'Hitmonlee',
 'Horsea',
 'Hypno',
 'Ivysaur',
 'Jigglypuff',
 'Jolteon',
 'Jynx',
 'Kabuto',
 'Kabutops',
 'Kadabra',
 'Kakuna',
 'Kangaskhan',
 'Kingler',
 'Koffing',
 'Krabby',
 'Lapras',
 'Lickitung',
 'Machamp',
 'Machoke',
 'Machop',
 'Magikarp',
 'Magmar',
 'Magnemite',
 'Magneton',
 'Mankey',
 'Marowak',
 'Meowth',
 'Metapod',
 'Mew',
 'Mewtwo',
 'Moltres',
 'MrMime',
 'Muk',
 'Nidoking',
 'Nidoqueen',
 'Nidorina',
 'Nidorino',
 'Ninetales',
 'Oddish',
 'Omanyte',
 'Omastar',
 'Onix',
 'Paras',
 'Parasect',
 'Persian',
 'Pidgeot',
 'Pidgeotto',
 'Pidgey',
 'Pikachu',
 'Pinsir',
 'Poliwag',
 'Poliwhirl',
 'Poliwrath',
 'Ponyta',
 'Porygon',
 'Primeape',
 'Psyduck',
 'Raichu',
 'Rapidash',
 'Raticate',
 'Rattata',
 'Rhydon',
 'Rhyhorn',
 'Sandshrew',
 'Sandslash',
 'Scyther',
 'Seadra',
 'Seaking',
 'Seel',
 'Shellder',
 'Slowbro',
 'Slowpoke',
 'Snorlax',
 'Spearow',
 'Squirtle',
 'Starmie',
 'Staryu',
 'Tangela',
 'Tauros',
 'Tentacool',
 'Tentacruel',
 'Vaporeon',
 'Venomoth',
 'Venonat',
 'Venusaur',
 'Victreebel',
 'Vileplume',
 'Voltorb',
 'Vulpix',
 'Wartortle',
 'Weedle',
 'Weepinbell',
 'Weezing',
 'Wigglytuff',
 'Zapdos',
 'Zubat']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': str(learn.predict(img)[0])})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)
