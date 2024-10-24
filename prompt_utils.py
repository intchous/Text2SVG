import random
from itertools import product


def get_negtive_prompt_text():

    # https://www.douban.com/people/166728879/?dt_dapp=1&_i=1385261JCWg0Nz
    BASE_PROMPT = ",(((lineart))),((low detail)),(simple),high contrast,sharp,2 bit"
    # BASE_NEGPROMPT = "(((text))),((color)),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error,Illustration, watermark"
    BASE_NEGPROMPT = "(((text))),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error, watermark, stripe"

    neg_prompt = "lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"

    BASE_NEGPROMPT = BASE_NEGPROMPT + ", " + neg_prompt

    StyleDict = {
        "Illustration": BASE_PROMPT + ",(((vector graphic))),medium detail",

        "Logo": BASE_PROMPT + ",(((centered vector graphic logo))),negative space,stencil,trending on dribbble",

        "Drawing": BASE_PROMPT + ",(((cartoon graphic))),childrens book,lineart,negative space",

        "Artistic": BASE_PROMPT + ",(((artistic monochrome painting))),precise lineart,negative space",

        "Tattoo": BASE_PROMPT + ",(((tattoo template, ink on paper))),uniform lighting,lineart,negative space",

        "Gothic": BASE_PROMPT +
        ",(((gothic ink on paper))),H.P. Lovecraft,Arthur Rackham",

        "Anime": BASE_PROMPT + ",(((clean ink anime illustration))),Studio Ghibli,Makoto Shinkai,Hayao Miyazaki,Audrey Kawasaki",

        "Cartoon": BASE_PROMPT + ",(((clean ink funny comic cartoon illustration)))",

        "Sticker": ",(Die-cut sticker, kawaii sticker,contrasting background, illustration minimalism, vector, pastel colors)",

        "Gold Pendant": ",gold dia de los muertos pendant, intricate 2d vector geometric, cutout shape pendant, blueprint frame lines sharp edges, svg vector style, product studio shoot", "None - prompt only": ""
    }

    return BASE_NEGPROMPT


def get_prompt_text(signature, description_start="a clipart of ", description_end=" "):

    prompt_appd = "(((vector graphic))),(simple),(((white background))),((no outlines)), minimal flat 2d vector icon. lineal color. illustation, cartoon, 2d, clear, svg vector style, best quality, trending on artstation. trending on dribbble, masterpiece, beautiful detailed, cute, high resolution, intricate detail, hignity 8k wallpaper, detailed"

    prompt_appd = "(((vector graphic))),(simple),(((white background))),((no outlines)), minimal flat 2d vector icon. lineal color. trending on artstation. professional vector illustration, cartoon, clear, 2d, svg vector style, best quality. trending on dribbble. masterpiece, beautiful detailed, cute, high resolution, intricate detail, hignity 8k wallpaper, detailed"

    prompt = description_start + signature + description_end + prompt_appd

    return prompt


def get_prompt_description_vectorfusion_list():
    prompt_description_vectorfusion_list = [
        # "a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
        # "a photograph of a fiddle next to a basketball on a ping pong table",
        # "a basketball to the left of two soccer balls on a gravel driveway",
        # "a bottle of beer next to an ashtray with a half-smoked cigarette",
        "a tall horse next to a red car",
        "A dragon-cat hybrid",
        "A photograph of a cat",
        # "A realistic photograph of a cat",
        # "A cat as 3D rendered in Unreal Engine",
        "A Japanese woodblock print of one cat",
        "A watercolor painting of a cat",
        # "A 3D wireframe model of a cat",
        "a baby penguin",
        "a boat",
        "a ladder",
        "an elephant",
        "a tree",
        "a crown",
        "the Great Wall",
        "Horse eating a cupcake",
        # "A 3D rendering of a temple",
        "A temple",
        "Family vacation to Walt Disney World",
        "Underwater Submarine",
        "Forest Temple",
        # "Forest Temple as 3D rendered in Unreal Engine",
        "Watercolor painting of a fire-breathing dragon",
        "A spaceship flying in a starry sky",
        "A painting of a starry night sky",
        "Yeti taking a selfie",
        "Bicycle",
        "Fast Food",
        "A drawing of a cat",
        # "Third eye",
        # "Self",
        # "Happiness",
        # "Translation",
        # "The space between infinity",
        # "Enlightenment",
        # "Hashtag",
        # "A chihuahua wearing a tutu",
        # "A hotdog in a tutu skirt",
        # "a child unraveling a roll of toilet paper",
        # "yin-yang",
        # "molecule",
        "The Eiffel Tower",
        "A torii gate",
        "A realistic painting of a sailboat",
        "Landscape",
        "Walt Disney World",
        "A picture of Tokyo",
        "The Tokyo Tower is a communications and observation tower in the Shiba-koen district of Minato, Tokyo, Japan",
        "The United States of America, commonly known as the United States or America, is a country primarily located in North America",
        "The corporate headquarters complex of Google, located at 1600 Amphitheatre Parkway in Mountain View, California",
        "the Imperial State Crown of England",
        "the Great Pyramid",
        "a red robot and a black robot standing together",
        "a black dog jumping up to hug a woman wearing a red sweater",
        "a painting of the Mona Lisa on a white wall",
        "an owl standing on a wire",
        "A banana peeling itself",
        "A blue poison-dart frog sitting on a water lily",
        "A brightly colored mushroom growing on a log",
        "A bumblebee sitting on a pink flower",
        "A delicious hamburger",
        "A fox playing the cello",
        "An exercise bike",
        "An erupting volcano, aerial view",
        "An octopus playing the harp",
        "A pig wearing a backpack",
        "A snail on a leaf",
        "A roulette wheel",
        "A rabbit cutting grass with a lawnmower",
        "A tiger karate master",
        "A yellow schoolbus",
        "a bus covered with assorted colorful graffiti on the side of it",
        "The titanic, aerial view",
        "A smiling sloth wearing a leather jacket, a cowboy hat and a kilt",
        "A photo of a Ming Dynasty vase on a leather topped table",
        "an ornate metal bench by a nature path",
        "a stop sign with a large tree behind it",
        "a kids book cover with an illustration of brown dog driving a red pickup truck",
        "a chimpanzee wearing a bowtie and playing a piano",
        "a great gray owl with a mouse in its beak",
        "a bamboo ladder propped up against an oak tree",
        "a light shining on a giraffe in a street",
        "a colorful rooster",
        "a beach with a cruise ship passing by",
        "a red pawn attacking a black bishop",
        "a mountain stream with salmon leaping out of it",
        "the silhouette of an elephant on the full moon",
        "an old-fashioned phone next to a sleek laptop",
        "a motorcycle parked in an ornate bank lobby",
        "a group of penguins in a snowstorm",
        "a helicopter hovering over Times Square",
        "a prop plane flying low over the Great Wall",
        "a tennis court with a basketball hoop in one corner",
        "The Alamo with bright grey clouds above it",
        "a tuba with red flowers protruding from its bell",
        "a family of bears passing by the geyser Old Faithful",
        "a politician giving a speech at a podium",
        "a girl getting a kite out of a tree",
        "earth",
        "fire",
        "a shiba inu",
        "an espresso machine",
        "A bowl of Pho",
        "a cup of boba",
        "Golden Gate bridge on the surface of Mars",
        "A picture of some food in the plate",
        "a chair",
        "the Empire State Building",
        "the Sydney Opera House",
        "a hedgehog",
        "a panda rowing a boat in a pond",
        "a robot",
        "a clock",
        "a teapot",
        "a train",
        "a boat in the canals of Venice",
        "the grand canyon",
        "a horse",
        "a pumpkin",
        "a knight holding a long sword",
        "The Statue of Liberty with the face of an owl",
        "a city intersection",
        "a circular logo",
        "a margarita",
        "an armchair in the shape of an avocado",
        "a stork playing a violin",
        "a pirate ship landing on the moon",
    ]

    return prompt_description_vectorfusion_list


def get_prompt_description_gpt_list():
    prompt_description_gpt_list = [
        # People Actions
        "A woman dancing with a red dress in the moonlight.",
        "A chef tossing a pizza in a kitchen setting.",
        "A firefighter dousing flames with a powerful water stream.",
        "A scientist examining colorful chemicals in a lab.",
        "An artist painting a bright sunflower on a canvas.",
        "A gardener pruning a large, blooming rose bush.",
        "A violinist playing a melody on a moonlit stage.",
        "A teacher reading a fairy tale to a group of kids.",
        "A jogger sprinting along a cityscape at dawn.",
        "A photographer capturing the moment of a butterfly landing.",
        "A kid joyfully building a towering sandcastle.",
        "A magician pulling a rabbit out of a hat.",
        "A sailor navigating by the stars.",
        "A yoga practitioner meditating on a serene beach.",
        "A skateboarder doing a kickflip over a staircase.",
        "A nurse tenderly bandaging a child's knee.",
        "A pilot steering a small plane above the clouds.",
        "A mechanic fixing an old, classic car.",
        "A fisherman reeling in a big catch at sunrise.",
        "A baker icing a multi-layered chocolate cake.",

        # People Characteristics
        "A man with bright blue hair and sunglasses.",
        "An astronaut floating in space outside a spacecraft.",
        "A girl with a patchwork quilted dress and a sun hat.",
        "A boy with oversized headphones and a skateboard.",
        "A cowboy in a large hat and leather boots.",
        "A hiker with a backpack and walking stick.",
        "An explorer with a map and binoculars.",
        "A superhero in a mask and cape, poised for action.",
        "A punk rocker with a spiked mohawk.",
        "A king with a jeweled crown and scepter.",
        "A samurai with traditional armor and a katana.",
        "A ballerina in a tutu and ballet shoes.",
        "A chef in a white hat and apron, holding a spatula.",
        "A robot with sleek metallic limbs and glowing eyes.",
        "A medieval knight in shining armor, holding a shield.",
        "A pirate with a hook hand and a parrot.",
        "A Victorian lady in an elegant gown and parasol.",
        "A DJ with neon gloves and a mixing deck.",
        "A girl scout with badges and a green sash.",
        "A fairy with delicate wings and a wand.",

        # Animal Characteristics
        "A dragon-cat hybrid with scales and whiskers.",
        "A dog with a peacock's tail and bright plumage.",
        "An eagle-bear creature with sharp talons.",
        "A translucent jellyfish with rainbow hues.",
        "A parrot with multi-colored feathers and a crest.",
        "A seal with smooth, shiny fur and big eyes.",
        "A wolf with a winter coat and piercing eyes.",
        "A butterfly with unusually large and vibrant wings.",
        "A squirrel with a fluffy, oversized tail.",
        "An owl with a heart-shaped face and large wings.",
        "A snake with vibrant, hypnotic patterns.",
        "A rabbit with long, floppy ears and a bushy tail.",
        "A turtle with a mosaic-patterned shell.",
        "A kangaroo with a baby in its pouch.",
        "A penguin dressed in a tiny bow tie.",
        "An elephant with intricate, painted tusks.",
        "A giraffe with unique, starry spot patterns.",
        "A shark with sleek, streamlined fins.",
        "A hummingbird with iridescent feathers.",

        # Scenarios
        "A cozy cottage nestled in a snowy forest.",
        "A majestic waterfall cascading into a crystal-clear lake.",
        "A bustling city street corner with bright neon signs.",
        "A serene Zen garden with a stone path and cherry blossoms.",
        "A vibrant coral reef teeming with marine life.",
        "A quaint village with cobblestone streets and thatched roofs.",
        "A futuristic city skyline with hovering vehicles.",
        "A peaceful mountain vista with a rising sun.",
        "A lively carnival with a Ferris wheel and colorful tents.",
        "An ancient castle perched on a rocky cliff.",
        "A lush jungle with exotic birds and dense foliage.",
        "A moonlit beach with gentle waves and a hammock.",
        "A charming bakery with pastries and a coffee aroma.",
        "A quiet library filled with towering bookshelves.",
        "A spooky haunted mansion under a full moon.",
        "A sunlit vineyard with rows of grapevines.",
        "A rustic farm with barns and animals grazing.",
        "A bustling train station with an arriving locomotive.",
        "A historic museum with grand pillars and artifacts.",
        "An enchanting forest path with twinkling fairy lights.",
    ]

    return prompt_description_gpt_list


def get_prompt_description_liveskt_list():
    # captions used in the main paper
    prompt_description_liveskt_list = [
        "The penguin is shuffling along the ice terrain, taking deliberate and cautious step with its flippers outstretched to maintain balance.",
        "The two dancers are passionately dancing the Cha-Cha, their bodies moving in sync with the infectious Latin rhythm.",
        "The boxer ducking and weaving to avoid his opponent's punches, and to punch him back.",
        "The runner runs with rhythmic leg strides and synchronized arm swing propelling them forward while maintaining balance.",
        "The jazz saxophonist performs on stage with a rhythmic sway, his upper body sways subtly to the rhythm of the music.",
        "The goldenfish is gracefully moving through the water, its fins and tail fin gently propelling it forward with effortless agility.",
        "The crab scuttled sideways along the sandy beach, its pincers raised in a defensive stance.",
        "The ballerina is dancing.",
        "A galloping horse.",
        "The hypnotized cobra snake sways rhythmically, mesmerized, with a gentle and trance-like side-to-side motion, its hood slightly flared, under the influence of an entrancing stimulus.",
        "The lizard moves with a sinuous, undulating motion, gliding smoothly over surfaces using its agile limbs and tail for balance and propulsion.",
        "The eagle soars majestically, with powerful wing beats and effortless glides, displaying precise control and keen vision as it maneuvers gracefully through the sky.",
        "The kangaroo is jumping, crouching, gathering energy, then propelling itself forward with incredible force and agility, tucking its legs mid-air before landing gracefully on its hind legs with remarkable balance and coordination.",
        "The man sailing the boat, his hands deftly manipulate the oars, while his body shifts subtly to maintain balance, while his boat moves foraward in the river.",
        "The cat is playing.",
        "The biker is pedaling, each leg pumping up and down as the wheels of the bicycle spin rapidly, propelling them forward.",
        "The cheetah is running at high speeds in pursuit of prey.",
        "A hummingbird hovers in mid-air and sucks nectar from a flower.",
        "A dolphin swimming and leaping out of the water.",
        "A butterfly fluttering its wings and flying gracefully.",
        "A gazelle galloping and jumping to escape predators.",
        "The squirrel uses its dexterous front paws to hold and manipulate nuts, displaying meticulous and deliberate motions while eating.",
        "A gymnast flipping, tumbling, and balancing on various apparatuses.",
        "A martial artist executing precise and controlled movements in different forms of martial arts.",
        "A figure skater gliding, spinning, and performing jumps on ice skates.",
        "A surfer riding and maneuvering on waves on a surfboard.",
        "A basketball player dribbling and passing while playing basketball.",
        "A waving flag fluttering and rippling in the wind.",
        "A parachute descending slowly and gracefully after being deployed.",
        "A wind-up toy car, moving forward or backward when wound up and released.",
        "A windmill spinning its blades in the wind to generate energy.",
        "A ceiling fan rotating blades to circulate air in a room.",
        "A clock hands ticking and rotating to indicate time on a clock face.",
        "The wine in the wine glass sways from side to side.",
        "The airplane moves swiftly and steadily through the air.",
        "The spaceship accelerates rapidly during takeoff, utilizing powerful rocket engines.",
        "The flower is moving and growing, swaying gently from side to side.",
        "A clock hands ticking and rotating to indicate time.",

        # pwarp clipart
        "A man walks forward.",
        "A young man jumps up and down.",
        "A man stands with one leg raised, stretching his leg muscles.",
        "A young man is waving his arms to say hello.",
        "The runner runs with rhythmic leg strides and synchronized arm swing propelling them forward.",
        "A red cartoon monster is cheering and waving its arms",
        "A woman dancer is dancing the Cha-Cha.",
        "A woman dancer is dancing with her legs moving up and down, and waving her hands",
        "A young girl jumps up and down.",
        "A young girl is standing still, waving her arms to say goodbye.",
        "A woman in a green dress with black polka dots and black boots is dancing joyfully.",
        "A woman in a flowing dance move, squatting slightly with her legs, wearing a red top, blue skirt, and black ankle boots.",
        "A bearded man is dancing with the music, his legs squatting up and down slightly, his hands waving in the air.",
        "A person wearing a pumpkin mask, pink tutu, and orange socks is in an energetic dance with arms raised.",
        "An elderly man is lifting dumbbells up and down in a rhythmic motion.",
        "An elderly woman with white hair is squatting up and down dramatically. Her hands are in a horizontal position to keep balance.",
        "A ninja is performing a high kick, with one leg moved up and the other leg bent. His hands are in a fighting position.",
        "A fencer in en garde position, ready to advance.",
        "A woman standing while holding a book in her hand.",
        "A woman is taking a selfie with her phone while waving the hand that holds an ice cream cone. Her feet are stationary.",
        "A woman is practicing yoga, bending her torso back and forth while extending her legs vertically.",
        "A seal is floating up and down in the water, waving its flippers and tail",
        "A turtle is swimming from left to right.",
        "A dolphin swimming and leaping out of the water.",
        "A crab is waving its pincers and legs continuously.",
        "A shrimp is swmming and swaying its tentacles.",
        "A flower is swaying in the wind.",
        "A flower sways its petals and leaves.",
        "A cloud floats up and down in the sky.",
        "A chicken is jumping up and down.",
        "a cheerful green caterpillar is arching and contracting its body.",
        "A bird is flying up and down.",
        "Boxing guy is punching and dodging.",
        "A starfish is waving its tentacles softly.",
        "An elephant jumps and wags its trunk up and down continuously.",
        "A kite is floating in the sky.",
        "A balloon floatings in the air.",
    ]
    return prompt_description_liveskt_list


def get_prompt_description_animatesvg_list():
    prompt_description_animatesvg_list = [
        "A man walks forward.",
        "A young man jumps up and down.",
        "A man stands with one leg raised, stretching his leg muscles.",
        "A young man is waving his arms to say hello.",
        "The runner runs with rhythmic leg strides and synchronized arm swing propelling them forward.",
        "A red cartoon monster is cheering and waving its arms",
        "A woman dancer is dancing the Cha-Cha.",
        "A woman dancer is dancing with her legs moving up and down, and waving her hands",
        "A young girl jumps up and down.",
        "A young girl is waving her arms to say goodbye.",
        "A woman in a green dress with black polka dots and black boots is dancing joyfully.",
        "A woman in a flowing dance move, squatting slightly with her legs, wearing a red top, blue skirt, and black ankle boots.",
        "A bearded man is dancing with the music, his legs squatting up and down slightly, his hands waving in the air.",
        "A person wearing a pumpkin mask, pink tutu, and orange socks is in an energetic dance with arms raised.",
        "An elderly man is lifting dumbbells up and down in a rhythmic motion.",
        "An elderly woman with white hair is squatting up and down dramatically. Her hands are in a horizontal position to keep balance.",
        "A ninja is performing a high kick, with one leg moved up and the other leg bent. His hands are in a fighting position.",
        "A fencer in en garde position, ready to advance.",
        "A woman standing while holding a book in her hand.",
        "A woman is taking a selfie with her phone while waving the hand that holds an ice cream cone. Her feet are stationary.",
        "A woman is practicing yoga, bending her torso back and forth while extending her legs vertically.",
        "A seal is floating up and down in the water, waving its flippers and tail",
        "A turtle floats up and down in the water, extending and retracting its legs.",
        "A dolphin swimming and leaping out of the water, bending its body flexibly.",
        "A crab is waving its pincers and legs continuously.",
        "A shrimp is swmming and swaying its tentacles.",
        "A flower sways its petals in the breeze.",
        "A flower sways its petals and leaves.",
        "A cloud floats in the sky.",
        "A chicken is jumping up and down.",
        "a cheerful green caterpillar is arching and contracting its body.",
        "A bird is flying up and down.",
        "Boxing guy is punching and dodging.",
        "A starfish is waving its tentacles softly.",
        "An elephant jumps and wags its trunk up and down continuously.",
        "A kite is floating in the sky.",
        "A balloon floating in the air.",
        "A man playing table tennis is swinging the racket.",
        "A man is scuba diving and swaying fins.",
        "A diver in mid-air, bending his body and legs to dive into the water.",
        "A surfer riding and maneuvering on waves on a surfboard.",
        "The ballerina is dancing, her arms bending and stretching up and down gracefully.",
        "A butterfly fluttering its wings and flying gracefully.",
        "A lemur is bending its long tail continuously.",
        "A black leopard is standing and shaking its tail.",
        "A stork bends its long neck.",
        "A candle with flickering flame.",
        "A bat is flapping its wings up and down.",
        "A jellyfish is floating up and down in the water, speading and swaying its tentacles.",
        "An octopus is swimming and swaying its tentacles.",
        "A snowman is waving its arms cheerfully.",
        "A bird is flappping its wings to make balance in the air.",
        "A black is playing and waving its long tail from left to right.",
        "A bat is flapping its wings up and down.",
        "The fish is gracefully moving through the water, its fins and tail fin gently propelling it forward with effortless agility.",
        "A dog is running.",
        "A spider sways its legs.",
        "A flamingo is walking.",
        "The raccoon is playing.",
        "The worm is arching and contracting its body.",
        "The man in demon costume is cheering, waving both arms up and down.",
        "The girl speaks into a megaphone",
        "A young girl is exercising in lunging position, lifting dumbbells.",
        "A galloping dog",
        "The spaceship accelerates rapidly during takeoff, flies into the sky.",
        "The ghost is dancing",
        "A burning flame sways from side to side.",
        "The cloud floats in the sky.",
        "A waving flag fluttering and rippling in the wind.",
        "The palm tree sways the leaves in the wind.",
        "A dragonfly is flappping its wings to make balance in the air.",
        "A snail is moving.",
        "A dolphin swimming and leaping out of the water, bending its body flexibly.",
        "The parrot flapping its wings.",
        "The firecracker flies into the sky.",
        "A robot is dancing",
        "A yello creature is dancing.",
        "A parachute descending slowly and gracefully after being deployed.",
        "The flower is moving and growing, swaying gently from side to side.",
        "The Halloween ghost is cheering and waving its arms.",
        "The gingerbread man is dancing.",
        "A woman is talking on the phone.",
        "The boxer is punching.",
        "Breakdancer performing an inverted freeze with one hand on the floor and legs artistically intertwined in the air.",
        "A rapper is singing, moving his hands and body to the rhythm of the music.",
        "A man is skiing down the slope.",
        "A bird hovers in mid-air, flapping wings energetically.",
        "A camel is walking.",
        "A man is eating, moving his hand to his mouth.",
        "A woman is swimming, bending her arms and legs continuously.",
        "A woman is doing gymnastics, the ribbon flowing flexibly in the air.",
    ]

    return prompt_description_animatesvg_list


# ---------------------------------------

def get_prompt_description_att3d_list(prompt_animals_list=[], do_extend=True):
    if (prompt_animals_list == []):
        prompt_animals_list = ["a squirrel", "a raccoon", "a pig", "a monkey",
                               "a robot", "a lion", "a rabbit", "a tiger", "an orangutan", "a bear"]
        if (do_extend):
            prompt_animals_list.extend(["a fox", "an elephant", "a zebra", "a rhino", "a dog", "a cat", "a parrot", "a penguin", "a leopard",
                                       "a dolphin", "a rat", "a crocodile", "a giraffe", "a wolf", "an ostrich", "an owl", "an antelope", "a bat", "a donkey"])

    prompt_activities_list = ["riding a motorcycle", "sitting on a chair", "playing the guitar",
                              "holding a shovel", "holding a blue balloon", "holding a book", "wielding a katana"]
    if (do_extend):
        prompt_activities_list.extend(["", "dancing", "riding a bicycle", "cooking pasta", "rowing a boat", "sleeping in bed", "running",
                                       "swimming", "painting a landscape outdoors", "picking fruits", "flying a kite", "playing video games", "making pottery", "ice skating"])

    prompt_themes_list = ["wearing a leather jacket", "wearing a sweater",
                          "wearing a cape", "wearing medieval armor", "wearing a backpack", "wearing a suit"]
    if (do_extend):
        prompt_themes_list.extend(["", "wearing an evening gown", "wearing a swimsuit", "wearing sportswear", "wearing a spacesuit", "wearing cowboy attire", "wearing a loose shirt", "wearing a robe", "wearing dance attire",
                                   "wearing a chef's uniform", "wearing a detective coat", "wearing casual clothes", "wearing pirate attire", "wearing overalls",  "wearing a cheongsam", "wearing a raincoat", "wearing a school uniform"])

    prompt_hats_list = ["wearing a party hat", "wearing a sombrero",
                        "wearing a helmet", "wearing a top hat", "wearing a baseball cap"]
    if (do_extend):
        prompt_hats_list.extend(["", "wearing a headscarf", "wearing a crown", "wearing earmuffs", "wearing sunglasses", "wearing a mask", "wearing a flower crown", "wearing a wizard hat", "wearing a pilot cap", "wearing a cowboy hat", "wearing a headband", "wearing a sports cap", "wearing a ski mask", "wearing a detective cap",
                                "wearing a country hat", "wearing pilot glasses"])

    # Generate all possible combinations of the lists
    prompt_combinations = list(product(
        prompt_animals_list, prompt_activities_list, prompt_themes_list, prompt_hats_list))

    # Create a list of formatted strings from combinations
    prompts_list = [f"{animal} {activity} {theme} {hat}" for animal,
                    activity, theme, hat in prompt_combinations]

    return prompts_list


# ---------------------------------------
def load_samp_prompts(dataset_name="vectorfusion", do_extend=True):
    if (dataset_name == "pig"):
        ini_samp_prompts = get_prompt_description_att3d_list(
            prompt_animals_list=["a pig"], do_extend=do_extend)
        random.shuffle(ini_samp_prompts)
        samp_prompts = ini_samp_prompts

    elif (dataset_name == "animal"):
        ini_samp_prompts = get_prompt_description_att3d_list(
            prompt_animals_list=[], do_extend=do_extend)
        random.shuffle(ini_samp_prompts)
        samp_prompts = ini_samp_prompts

    elif (dataset_name == "animatesvg"):
        # ini_samp_prompts = get_prompt_description_liveskt_list()
        ini_samp_prompts = get_prompt_description_animatesvg_list()
        random.shuffle(ini_samp_prompts)

        small_prompts_list = ["The runner runs with rhythmic leg strides and synchronized arm swing propelling them forward.", "A woman dancer is dancing the Cha-Cha.", "A woman in a green dress with black polka dots and black boots is dancing joyfully.", "A woman dancer is dancing with her legs moving up and down, and waving her hands",
                              "A woman in a flowing dance move, squatting slightly with her legs, wearing a red top, blue skirt, and black ankle boots.", "A bearded man is dancing with the music, his legs squatting up and down slightly, his hands waving in the air.", "A person wearing a pumpkin mask, pink tutu, and orange socks is in an energetic dance with arms raised."]

        random.shuffle(small_prompts_list)
        small_animatesvg_list = small_prompts_list
        small_animatesvg_list.extend(ini_samp_prompts)

        samp_prompts = small_animatesvg_list

    elif (dataset_name == "vectorfusion"):
        vecf_samp_prompts = get_prompt_description_vectorfusion_list()
        random.shuffle(vecf_samp_prompts)

        ini_samp_prompts = get_prompt_description_gpt_list()
        # ini_samp_prompts = get_prompt_description_liveskt_list()
        # ini_samp_prompts = get_prompt_description_animatesvg_list()
        random.shuffle(ini_samp_prompts)

        samp_prompts = ["A lake with trees and mountains in the background, teal sky", "A painting of a Chinese temple with mountains in the background", "A castle stands in front of the mountains", "A poster of the great wall, teal and orange color scheme, autumn colors", "Orange cute watercolor fox", "Avocados", "A portrait of an astronaut. the logo, MS_emoji_style", "A portrait of an astronaut. the logo, MS_emoji_style, Van Gogh style", "Pikachu, in pastel colors, childish and fun", "Darth vader", "A picture of a polar bear", "A picture of a tiger", "A picture of a macaw", "The image captures the essence of Vincent van Gogh, colorful world he painted",
                        "Big Wild Goose Pagoda", "a man in an astronaut suit walking across a desert, planet mars in the background", "Full image of a Japanese sakura tree on a hill", "Sydney opera house,oil painting,by Van Gogh", "A photograph of an astronaut riding a horse", "A colorful German shepherd in vector art", "A phoenix coming out of the fire drawing", "Watercolor landscape painting ancient villages", "A photo of a futuristic robot", "An owl stands on a branch", "The logo of the Japanese mystery temple,game art,cartoon,animation style", "A Starbucks coffee cup", "a SPACEMAN", "a Superhero"]

        samp_prompts.extend(ini_samp_prompts)
        samp_prompts.extend(vecf_samp_prompts)
        random.shuffle(samp_prompts)

        style_list = ["minimal flat 2d vector icon", "the logo", "pop art style", "Anime style", "Pixar style", "Vintage style", "Animal Crossing style", "Miyazaki Hayao style", "Makoto Shinkai animation style",
                      "Ghibli Studio style", "line art style", "watercolor painting", "Traditional Chinese Painting", "Tradition Chinese Ink Painting", "Ink Wash Painting Style", "Cyberpunk", "Hollywood style", "Ukiyo-e art"]

        small_prompts_list = ["a cute dinosaur", "A portrait of a cute clown with a clown hat", "The Girl with a Pearl Earring", "The Grand Budapest Hotel", "A girl with sunglasses", "A cat",
                              "A magician pulling a rabbit out of a hat", "A girl with dress and a sun hat", "A castle stands in front of the mountain", "The Japanese mystery temple", "The logo of the GamePad"]

        samp_prompts.extend(small_prompts_list)
        random.shuffle(samp_prompts)

        style_list = ["Anime style", "Miyazaki Hayao style",
                      "Makoto Shinkai animation style"]

        small_prompts_list = ["A portrait of a cat", "A girl with sunglasses", "A girl with a hat",
                              "The Girl with a Pearl Earring", "A portrait of a clown with a cute clown hat"]

        # --------------------------------------------------
        small_sty_list = [
            f"{prompt}, {style}" for prompt in small_prompts_list for style in style_list]

        small_prompts_list = small_sty_list
        random.shuffle(small_prompts_list)
        # --------------------------------------------------

        small_prompts_list.extend(samp_prompts)
        samp_prompts = small_prompts_list

    return samp_prompts


# ---------------------------------------
def get_prompt_description_face_list():
    prompt_description_face_list = [
        "wearing a cowboy hat",
        "wearing hat flower",
        "wearing Birthday Hat",
        "wearing crown",
        "wearing hair scrunchie",
        "wearing top hat",
        "wearing headphones",
        "wearing earrings",

        "short hair",
        "long hair",
        "blonde hair",
        "blue hair",
        "wavy hair",

        "with hair bell",
        "with hair ornament",
        "with horns",
        "with cat ears",

        "wearing neckwear",

        "hair over one eye",
        "with blue eyes",
        "with sparkling eyes",

        "opening mouth",
        "with tongue out",

        "bursting into tears",
        "smiling and happy",
        "angry",

        "",

    ]

    return prompt_description_face_list


def get_prompt_description_animal_list():
    prompt_description_animal_list = [
        "bursting into tears",
        "smiling and happy",
        "angry",

        "wearing a cowboy hat",
        "wearing hat flower",
        "wearing Birthday Hat",
        "wearing crown",
        # "wearing hair scrunchie",
        "wearing top hat",
        # "wearing headphones",
        # "wearing earrings",

        "with fox tail",
        "with dragon tail",

        "opening mouth",
        "with tongue out",

        "driving a red pickup truck",
        "raising a golden trophy",
        "rowing a boat",

        "taking photos",
        "taking a selfie",

        "under an umbrella",
        "playing the cello",

        "listening to music",
        "reading a book",
        "playing a ball",
        "going at skateboarding",

        "holding coffee",
        "holding laptop",
        "carrying a bag",

        "jumping",
        "running",

        "holding flower",
        "holding a cake",
        "salute",

        "standing on a tree stump",

        "talking on phone",

        "with a bird standing on the top",
        "",

    ]

    return prompt_description_animal_list


def get_prompt_description_scene_list():
    prompt_description_scene_list = [
        "in Autumn with fallen leaves",
        "snow mountain",
        "in the desert with cactus",

        "with ocean",
        "near beach",
        "with volcano",
        "with geyser",

        "with starry sky",
        "cloudy",
        "with sun",

        "",

    ]

    return prompt_description_scene_list


def get_prompt_description_action_list():
    prompt_description_action_list = [
        "raising a golden trophy",
        "rowing a boat",

        "taking photos",
        "taking a selfie",

        "under an umbrella",
        "playing the cello",
        "walking a dog",

        "listening to music",
        "reading a book",
        "playing a ball",
        "going at skateboarding",

        "holding coffee",
        "holding laptop",
        "carrying a bag",
        "wearing a backpack",
        "wearing a cowboy hat",
        "holding flower",

        "jumping",
        "running",

        "salute",

        "talking on phone",

        "sitting at the table",

        "wearing boots",
        "wearing high heels",

        "with a bird standing on the top",
        "",
    ]

    return prompt_description_action_list
