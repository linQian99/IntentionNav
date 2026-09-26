"""Fixed visual attributes for all 64 classes; offline R073 candidate only.

Descriptions express general category knowledge and contain no episode IDs,
positions, room predictions or instance metadata. Every class has equal weight.
They are soft text similarities, not necessary conditions or new goal aliases.
"""
from __future__ import annotations


VISUAL_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "air conditioner": ("a vented appliance with long air outlet slats", "a box shaped unit with an intake grille and air louvers"),
    "air purifier": ("an appliance with a perforated air intake grille", "a compact upright housing with an air outlet and controls"),
    "basin": ("a concave wash bowl with a drain opening", "a sink with a curved inner surface and a raised rim"),
    "bathtub": ("a large elongated bathing basin", "a deep hollow tub with broad sides and a rim"),
    "bed": ("a wide mattress on a bed frame", "a raised sleeping surface with bedding and a headboard"),
    "bench": ("a long seat supported by legs", "an elongated seating surface for more than one person"),
    "book": ("a bound stack of paper pages with a cover", "a rectangular volume with a spine and page edges"),
    "bowl": ("a deep round open container", "a curved vessel with an open top and a raised rim"),
    "bread machine": ("a countertop bread maker with a hinged top lid", "a box shaped baking appliance with buttons and a lid"),
    "builtin oven": ("a built in cooking appliance with a glass front door", "a rectangular oven door with a handle and control panel"),
    "cabinet": ("an enclosed storage body with doors or drawers", "a cupboard with framed panels and handles"),
    "ceiling light": ("a light fixture attached to or suspended from a ceiling", "an overhead luminaire with lamps or diffusers"),
    "chair": ("a single seat with legs and a backrest", "a seating frame with a seat and an upright back"),
    "chopstick": ("a pair of thin straight eating sticks", "long narrow rods with tapered tips"),
    "clock": ("a time display with hands or digits", "a clock face with hour markings or a digital time display"),
    "coffee maker": ("a coffee brewing appliance with a dispensing outlet", "a countertop coffee machine with a reservoir and controls"),
    "computer monitor": ("a computer display screen on a desk stand", "a flat rectangular screen with a bezel and support base"),
    "cosmetic": ("makeup or skincare bottles and containers", "small packaged beauty products in jars tubes or bottles"),
    "cup": ("a small drinking vessel with an open top", "a hollow container with upright sides and a rim"),
    "curtain": ("hanging fabric panels with vertical folds", "draped cloth suspended along its upper edge"),
    "desk": ("a work surface with space underneath for a chair", "a writing table with a flat top and legroom"),
    "dining table": ("a broad tabletop supported at seated eating height", "a table with space around the top for several place settings"),
    "dish washer": ("a dishwashing appliance with a broad hinged front door", "a rectangular dishwasher panel with a handle and controls"),
    "electric cooker": ("an electric cooking pot with a fitted lid", "a countertop rice cooker with a round inner pot and controls"),
    "floor lamp": ("a tall freestanding lamp with a floor base", "a light shade supported by a long upright stand"),
    "flower": ("a blossom with petals around a center", "flower heads on stems with leaves"),
    "fork": ("an eating utensil with several narrow prongs", "a long handle ending in a row of pointed tines"),
    "fridge": ("a large refrigerator cabinet with insulated doors", "a tall food cooling appliance with broad door panels"),
    "fruit": ("whole edible fruits with skin or peel", "rounded or elongated produce such as apples oranges and bananas"),
    "kettle": ("a water vessel with a handle lid and pouring spout", "a covered kettle body with a curved handle and short spout"),
    "knife": ("a cutting blade attached to a handle", "an elongated utensil with a flat sharp edged blade"),
    "menorah": ("a branched candle holder with multiple candle sockets", "a candelabrum with arms extending from a central stem"),
    "microwave": ("a countertop oven with a front door and side controls", "a box shaped microwave oven with a dark viewing window"),
    "mirror": ("a reflective glass surface", "a framed or unframed panel reflecting nearby objects"),
    "night stand": ("a small low bedside table", "a compact side table with a top and drawers or shelves"),
    "painting": ("a picture made with painted colors and brushwork", "a painted artwork on a canvas or panel"),
    "piano": ("a musical instrument with black and white keys", "a piano keyboard set into a large upright or grand body"),
    "picture frame": ("a border surrounding a displayed photograph or picture", "a rectangular or shaped frame enclosing an image"),
    "pillow": ("a soft stuffed cushion in a fabric cover", "a plump fabric pad with rounded edges"),
    "plate": ("a shallow flat dish with a raised edge", "a broad eating dish with a low rim"),
    "range hood": ("a wide cooking extractor hood with an underside vent", "a canopy shaped ventilation appliance above a cooking area"),
    "reed diffuser": ("a fragrance bottle with thin reeds protruding upward", "a small open bottle holding several diffuser sticks"),
    "remote control": ("a small handheld controller with many buttons", "an elongated control device with a keypad"),
    "room divider": ("a freestanding partition made of tall panels", "a folding screen with connected upright sections"),
    "sculpture": ("a three dimensional carved or modeled artwork", "a decorative solid figure or abstract art object"),
    "shelf": ("open horizontal surfaces supporting stored objects", "a shelving structure with accessible tiers or ledges"),
    "sofa": ("an upholstered seat with a back and space for several people", "a couch with padded cushions and armrests"),
    "sound": ("an audio loudspeaker with a grille or speaker drivers", "a speaker enclosure with circular sound producing cones"),
    "spoon": ("an eating utensil with a shallow oval bowl and handle", "a long narrow handle ending in a rounded scooping head"),
    "stool": ("a small seat without a tall backrest", "a compact seating surface supported by legs or a pedestal"),
    "storage": ("an organizer or container for stored items", "a storage unit with compartments baskets or bins"),
    "table": ("a flat raised horizontal surface supported by a base", "a tabletop with legs or a central pedestal"),
    "table lamp": ("a compact lamp with a short stand and base", "a light shade supported above a tabletop sized base"),
    "tea set": ("a coordinated group of teacups and tea serving vessels", "a tea service with cups saucers and a teapot"),
    "television": ("a large flat television display with a narrow bezel", "a broad screen for viewing television programs"),
    "toilet": ("a sanitary bowl with a seat and lid", "a toilet bowl with a pedestal and flushing unit"),
    "towel rack": ("bars or rails for hanging towels", "a frame with horizontal rods over which towels can be draped"),
    "toy": ("a plaything such as a doll or toy vehicle", "a small object shaped for play such as a stuffed animal or model"),
    "vase": ("an open decorative vessel with a neck and hollow body", "a container shaped to hold cut flower stems"),
    "wall light": ("a light fixture mounted on a wall bracket", "a wall sconce with a shade or exposed light source"),
    "washing machine": ("a laundry appliance with a drum opening", "a washing machine body with a loading door and controls"),
    "water cooler": ("a drinking water dispenser with taps", "an upright dispensing appliance with a recessed cup area"),
    "water heater": ("a water heating appliance with pipe connections", "a tank or wall mounted heater housing with controls"),
    "wine set": ("a coordinated group of wine serving glasses and vessels", "a wine service with stemmed glasses and a bottle or decanter"),
}


def category_prompts(label: str) -> tuple[str, str, str]:
    """Return one canonical-name prompt and two equally weighted attributes."""
    first, second = VISUAL_ATTRIBUTES[label]
    return (f"a photo of a {label}",
            f"a photo of a {label}, showing {first}",
            f"a photo of a {label}, showing {second}")
