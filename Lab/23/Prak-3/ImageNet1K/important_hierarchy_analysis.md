# Important Hierarchy Analysis

This note summarizes which ImageNet/WordNet hierarchy branches are most important to inspect for future projects that may use ImageNet-pretrained backbones.

The main question this analysis helps answer is:

- how likely is ImageNet pretraining to transfer well to a new downstream dataset?

The practical answer usually depends on:

- how close the downstream dataset is to an existing ImageNet semantic branch
- how many ImageNet classes exist under that branch
- how fine-grained those classes are
- how visually consistent the branch is

## Why Hierarchy Matters

ImageNet pretraining is not uniformly useful across all possible tasks.

A pretrained model tends to transfer best when:

- the target dataset lies near a well-represented ImageNet branch
- ImageNet contains many visually related subclasses
- the pretrained backbone has already learned strong discriminative features in that region

So the most important hierarchy branches are the ones that:

- have many classes
- are visually rich
- appear often in real-world computer vision projects

## Most Important Branches

## 1. Animal

This is one of the highest-value branches for transfer learning.

Important sub-branches:

- `entity > physical entity > object > whole > living thing > organism > animal`
- `animal > chordate > vertebrate > mammal`
- `animal > bird`
- `animal > fish`
- `animal > reptile`
- `animal > primate`
- `animal > hoofed mammal`

Why it matters:

- many practical image-classification tasks are animal-related
- animal categories often have strong visual structure
- ImageNet includes both broad and fine-grained animal supervision

### Especially Important Animal Sub-branches

#### Dog / Canine

Important chain:

- `animal > chordate > vertebrate > mammal > placental > carnivore > canine > dog`

Why it matters:

- ImageNet contains many dog breeds
- pretrained models usually learn very strong dog-related features
- transfer to dog or dog-adjacent tasks is usually strong

#### Cat / Feline

Important chain:

- `animal > chordate > vertebrate > mammal > placental > carnivore > feline > cat`

Why it matters:

- cat-related transfer is still useful
- but ImageNet has far fewer domestic-cat classes than dog classes
- for cat-vs-dog or feline-heavy tasks, pretrained features may be somewhat asymmetric

#### Bird

Important chain:

- `animal > chordate > vertebrate > bird`

Why it matters:

- ImageNet has many bird classes
- bird classification often benefits from pretrained fine-grained visual features
- transfer is usually strong for species-level or family-level bird tasks

#### Fish / Aquatic Vertebrate

Important chain:

- `animal > chordate > vertebrate > aquatic vertebrate > fish`

Why it matters:

- ImageNet includes a meaningful fish branch
- useful for marine-biology, aquarium, fisheries, and underwater classification tasks
- transfer can be good when the downstream task still uses full-animal visual cues rather than only tiny local patterns

#### Reptile

Important chain:

- `animal > chordate > vertebrate > reptile`

Why it matters:

- relevant for snake, lizard, turtle, and crocodilian tasks
- ImageNet contains both broad reptile coverage and some fine-grained snake categories
- transfer can be strong when the target dataset is close to these visual forms

#### Amphibian

Important chain:

- `animal > chordate > vertebrate > amphibian`

Why it matters:

- smaller branch than mammals or birds
- still useful for frog, salamander, and related animal tasks
- transfer is possible, but the branch is usually not as rich as dog, bird, or fish

#### Primate

Important chain:

- `animal > chordate > vertebrate > mammal > placental > primate`

Why it matters:

- useful for monkey, ape, and lemur classification tasks
- ImageNet includes several primate categories
- transfer can work well for wildlife projects with similar species structure

#### Hoofed Mammal / Ungulate

Important chain:

- `animal > chordate > vertebrate > mammal > placental > ungulate`

Why it matters:

- relevant for livestock, zoo, safari, and wildlife datasets
- includes deer-like, sheep-like, goat-like, bovine, pig-like, camelid, and similar groups
- transfer can be strong when body shape and silhouette dominate

#### Insect

Important when future projects involve:

- insects
- pests
- agricultural monitoring

Why it matters:

- ImageNet contains many insect categories
- transfer can be good when shape and texture cues matter

#### Bear / Mustelid / Small Carnivorous Mammal

Important chain:

- `animal > chordate > vertebrate > mammal > placental > carnivore`

Why it matters:

- useful for bear, otter, ferret, badger, skunk, and related tasks
- this branch is not as deep as dog, but it still provides meaningful carnivore-side supervision
- can help for wildlife datasets where the target classes are visually close to these animals

#### Rabbit / Rodent / Small Mammal

Important chain:

- `animal > chordate > vertebrate > mammal > placental > small mammal`

Why it matters:

- useful for rabbit, hamster, squirrel, beaver, guinea pig, and similar tasks
- transfer often helps when the downstream dataset focuses on body-shape and fur-pattern cues

#### Marine Invertebrate / Crustacean / Mollusk

Important chain:

- `animal > invertebrate > marine invertebrate`

Why it matters:

- useful for shellfish, crab, lobster, jellyfish, and similar recognition tasks
- weaker than the mammal branches for many standard CV tasks, but still valuable for specialized biology or seafood-related datasets

### Animal Branch Recommendation

For future animal-related projects, inspect these branches in this practical order:

1. `dog / canine`
2. `bird`
3. `cat / feline`
4. `fish / aquatic vertebrate`
5. `ungulate / hoofed mammal`
6. `reptile`
7. `primate`
8. `insect`
9. `small mammal`
10. `marine invertebrate`
11. `amphibian`

This ordering is not absolute, but it is a good practical rule because it combines:

- ImageNet branch richness
- typical transfer usefulness
- frequency of real-world downstream tasks

## 2. Artifact

This is one of the most useful branches for general computer vision projects.

Important chains include:

- `artifact > instrumentality`
- `artifact > device`
- `artifact > vehicle`
- `artifact > furniture`
- `artifact > tool`
- `artifact > container`
- `artifact > clothing`

Why it matters:

- many industrial and product-vision tasks are artifact-centric
- ImageNet has large and diverse artifact coverage
- transfer is often very strong for objects with stable shape cues

### Especially Important Artifact Sub-branches

#### Vehicle

Important chain:

- `artifact > instrumentality > conveyance > vehicle`

Why it matters:

- useful for transportation, robotics, traffic, and inspection tasks
- includes many visually distinct object families

#### Device / Instrument

Important chain:

- `artifact > instrumentality > device`

Why it matters:

- useful for electronics, hardware, medical-device-like objects, and general object recognition
- pretrained models often transfer well to this branch

#### Clothing / Wearables

Important for:

- fashion
- retail
- garment recognition
- assistive systems

Why it matters:

- ImageNet includes many wearable artifacts
- transfer is useful when shape and appearance cues dominate

## 3. Food

Important chain:

- `matter > substance > food`

Important sub-branches:

- `food > fruit`
- `food > prepared food`
- `food > baked goods`
- `food > beverage`
- `food > plant food / vegetables`

Why it matters:

- many applied ML projects involve food recognition
- pretrained models can transfer reasonably well here
- hierarchy helps estimate whether the source coverage is broad or sparse

## 4. Plant / Fungus

Important branch for:

- agriculture
- biology
- ecology
- crop recognition
- mushroom or plant classification

Important sub-branches:

- flowering plants
- seeds and fruiting structures
- fungi

Why it matters:

- transfer can still be useful even if ImageNet is not as strong here as in artifacts or dogs
- this branch is especially relevant for future crop- or species-oriented projects

## 5. Natural Scenes And Landforms

Important for:

- environment recognition
- geolocation-like tasks
- scene understanding
- terrain and landform classification

Important sub-branches:

- geological formations
- coasts and shores
- lakesides and natural landscapes

Why it matters:

- useful when the target task is scene-dominant rather than object-dominant
- transfer can help, but ImageNet is generally more object-centered than scene-centered

## 6. Human-Adjacent Categories

ImageNet is less ideal for human-centered projects than people sometimes assume.

Useful branches include:

- `person`
- some clothing and wearable branches
- some sports or object-use contexts

Why it matters:

- ImageNet is not primarily a human identity, action, or pose dataset
- for face recognition, gesture, pose, or activity tasks, transfer may be weaker or only indirectly useful

### Person

Important chain:

- `entity > physical entity > object > whole > living thing > organism > person`

Why it matters:

- this is the closest direct WordNet person branch inside ImageNet1k
- it is very small compared with animals or artifacts, with only `3` classes
- transfer can still help for coarse human-presence or sports-equipment context tasks
- it is usually not enough by itself for identity, pose, gesture, or rich action recognition

## What To Check For A Future Project

When evaluating whether ImageNet pretraining is likely to help, inspect these properties:

### 1. Branch Match

Ask:

- does the target dataset lie inside or near a strong ImageNet branch?

If yes:

- transfer is more promising

### 2. Subclass Richness

Ask:

- how many ImageNet classes are under that branch?

If many:

- the pretrained model likely has richer features there

Example:

- `dog` is much richer than `domestic cat`

### 3. Visual Consistency

Ask:

- do classes in the branch share stable visual cues?

Branches with strong transfer often have:

- consistent shape
- texture
- part structure

Examples:

- dogs
- birds
- vehicles
- tools

### 4. Label Granularity

Ask:

- is the downstream dataset broader or narrower than the ImageNet classes?

If downstream labels are broader:

- ImageNet pretraining often helps a lot

Example:

- many ImageNet dog breeds can help a simpler binary dog-vs-not-dog task

## Practical Priority Order

If you want a simple order of what usually matters most in future projects:

1. `animal`
2. `artifact`
3. `vehicle / device / tool` sub-branches
4. `food`
5. `plant / fungus`
6. `natural scene`
7. `human-adjacent`

This is not a rule for all domains, but it is a good practical starting point.

## Recommendation

For any future dataset, do this before deciding your fine-tuning strategy:

1. find the nearest WordNet/ImageNet branch
2. count how many ImageNet classes exist under that branch
3. compare sibling branches
4. inspect how fine-grained the source classes are
5. use that information to decide whether to:
   - freeze the backbone first
   - partially fine-tune
   - fully fine-tune
   - expect weak or strong transfer

## Short Conclusion

The most important hierarchy branches for future ImageNet-based projects are the ones that are both:

- semantically close to the target task
- richly represented in ImageNet

In practice, the strongest branches to inspect first are:

- animal
- artifact
- food
- plant/fungus
- natural scene

Within those, the most valuable branches are usually the visually rich and fine-grained ones, such as:

- dog / canine
- bird
- vehicle
- device
- tool

These branches are the best indicators of whether ImageNet pretraining is likely to transfer well to a future project.


## Supporting Tables

The sections below combine the important-branch summary with the label tables for each important sub-branch.

### Summary

| Branch | Matched Path Fragment | Class Count | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| Dog / Canine | > canine > dog > | 118 | 66302 | 5900 |
| Cat / Feline | > feline > | 13 | 7090 | 650 |
| Bird | > bird > | 52 | 27942 | 2600 |
| Fish / Aquatic Vertebrate | > fish > | 16 | 8004 | 800 |
| Reptile | > reptile > | 36 | 17819 | 1800 |
| Amphibian | > amphibian > | 8 | 4816 | 400 |
| Primate | > primate > | 20 | 10690 | 1000 |
| Ungulate / Hoofed Mammal | > ungulate > | 17 | 9260 | 850 |
| Insect | > insect > | 27 | 16269 | 1350 |
| Crustacean | > crustacean > | 9 | 4933 | 450 |
| Mollusk | > mollusk > | 6 | 3466 | 300 |
| Vehicle | > vehicle > | 67 | 35998 | 3350 |
| Device | > device > | 124 | 67299 | 6200 |
| Container | > container > | 48 | 25259 | 2400 |
| Clothing | > clothing > | 48 | 25918 | 2400 |
| Food Fruit | > edible fruit > | 10 | 5334 | 500 |
| Prepared Food / Dish | > dish > | 11 | 6172 | 550 |
| Beverage | > beverage > | 4 | 2149 | 200 |
| Person | > person > | 3 | 1625 | 150 |
| Fungus | > fungus > | 7 | 3656 | 350 |
| Geological Formation | > geological formation > | 10 | 7165 | 500 |

### Dog / Canine

Domestic dog branch. This is one of the richest and most transferable ImageNet branches.

- Class count: `118`
- Train image count: `66302`
- Val image count: `5900`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 151 | n02085620 | Chihuahua | 709 | 50 |
| 152 | n02085782 | Japanese_spaniel | 333 | 50 |
| 153 | n02085936 | Maltese_dog | 410 | 50 |
| 154 | n02086079 | Pekinese | 776 | 50 |
| 155 | n02086240 | Shih-Tzu | 587 | 50 |
| 156 | n02086646 | Blenheim_spaniel | 726 | 50 |
| 157 | n02086910 | papillon | 669 | 50 |
| 158 | n02087046 | toy_terrier | 263 | 50 |
| 159 | n02087394 | Rhodesian_ridgeback | 1040 | 50 |
| 160 | n02088094 | Afghan_hound | 393 | 50 |
| 161 | n02088238 | basset | 491 | 50 |
| 162 | n02088364 | beagle | 488 | 50 |
| 163 | n02088466 | bloodhound | 343 | 50 |
| 164 | n02088632 | bluetick | 437 | 50 |
| 165 | n02089078 | black-and-tan_coonhound | 296 | 50 |
| 166 | n02089867 | Walker_hound | 577 | 50 |
| 167 | n02089973 | English_foxhound | 374 | 50 |
| 168 | n02090379 | redbone | 677 | 50 |
| 169 | n02090622 | borzoi | 302 | 50 |
| 170 | n02090721 | Irish_wolfhound | 401 | 50 |
| 171 | n02091032 | Italian_greyhound | 421 | 50 |
| 172 | n02091134 | whippet | 387 | 50 |
| 173 | n02091244 | Ibizan_hound | 577 | 50 |
| 174 | n02091467 | Norwegian_elkhound | 465 | 50 |
| 175 | n02091635 | otterhound | 360 | 50 |
| 176 | n02091831 | Saluki | 442 | 50 |
| 177 | n02092002 | Scottish_deerhound | 636 | 50 |
| 178 | n02092339 | Weimaraner | 522 | 50 |
| 179 | n02093256 | Staffordshire_bullterrier | 829 | 50 |
| 180 | n02093428 | American_Staffordshire_terrier | 488 | 50 |
| 181 | n02093647 | Bedlington_terrier | 427 | 50 |
| 182 | n02093754 | Border_terrier | 446 | 50 |
| 183 | n02093859 | Kerry_blue_terrier | 381 | 50 |
| 184 | n02093991 | Irish_terrier | 492 | 50 |
| 185 | n02094114 | Norfolk_terrier | 594 | 50 |
| 186 | n02094258 | Norwich_terrier | 568 | 50 |
| 187 | n02094433 | Yorkshire_terrier | 834 | 50 |
| 188 | n02095314 | wire-haired_fox_terrier | 473 | 50 |
| 189 | n02095570 | Lakeland_terrier | 407 | 50 |
| 190 | n02095889 | Sealyham_terrier | 361 | 50 |
| 191 | n02096051 | Airedale | 531 | 50 |
| 192 | n02096177 | cairn | 480 | 50 |
| 193 | n02096294 | Australian_terrier | 541 | 50 |
| 194 | n02096437 | Dandie_Dinmont | 450 | 50 |
| 195 | n02096585 | Boston_bull | 397 | 50 |
| 196 | n02097047 | miniature_schnauzer | 299 | 50 |
| 197 | n02097130 | giant_schnauzer | 441 | 50 |
| 198 | n02097209 | standard_schnauzer | 450 | 50 |
| 199 | n02097298 | Scotch_terrier | 396 | 50 |
| 200 | n02097474 | Tibetan_terrier | 468 | 50 |
| 201 | n02097658 | silky_terrier | 401 | 50 |
| 202 | n02098105 | soft-coated_wheaten_terrier | 452 | 50 |
| 203 | n02098286 | West_Highland_white_terrier | 486 | 50 |
| 204 | n02098413 | Lhasa | 391 | 50 |
| 205 | n02099267 | flat-coated_retriever | 931 | 50 |
| 206 | n02099429 | curly-coated_retriever | 480 | 50 |
| 207 | n02099601 | golden_retriever | 1069 | 50 |
| 208 | n02099712 | Labrador_retriever | 571 | 50 |
| 209 | n02099849 | Chesapeake_Bay_retriever | 563 | 50 |
| 210 | n02100236 | German_short-haired_pointer | 493 | 50 |
| 211 | n02100583 | vizsla | 907 | 50 |
| 212 | n02100735 | English_setter | 706 | 50 |
| 213 | n02100877 | Irish_setter | 985 | 50 |
| 214 | n02101006 | Gordon_setter | 925 | 50 |
| 215 | n02101388 | Brittany_spaniel | 987 | 50 |
| 216 | n02101556 | clumber | 411 | 50 |
| 217 | n02102040 | English_springer | 477 | 50 |
| 218 | n02102177 | Welsh_springer_spaniel | 632 | 50 |
| 219 | n02102318 | cocker_spaniel | 477 | 50 |
| 220 | n02102480 | Sussex_spaniel | 341 | 50 |
| 221 | n02102973 | Irish_water_spaniel | 344 | 50 |
| 222 | n02104029 | kuvasz | 614 | 50 |
| 223 | n02104365 | schipperke | 349 | 50 |
| 224 | n02105056 | groenendael | 678 | 50 |
| 225 | n02105162 | malinois | 363 | 50 |
| 226 | n02105251 | briard | 294 | 50 |
| 227 | n02105412 | kelpie | 472 | 50 |
| 228 | n02105505 | komondor | 427 | 50 |
| 229 | n02105641 | Old_English_sheepdog | 413 | 50 |
| 230 | n02105855 | Shetland_sheepdog | 412 | 50 |
| 231 | n02106030 | collie | 484 | 50 |
| 232 | n02106166 | Border_collie | 1188 | 50 |
| 233 | n02106382 | Bouvier_des_Flandres | 612 | 50 |
| 234 | n02106550 | Rottweiler | 559 | 50 |
| 235 | n02106662 | German_shepherd | 901 | 50 |
| 236 | n02107142 | Doberman | 1162 | 50 |
| 237 | n02107312 | miniature_pinscher | 460 | 50 |
| 238 | n02107574 | Greater_Swiss_Mountain_dog | 682 | 50 |
| 239 | n02107683 | Bernese_mountain_dog | 458 | 50 |
| 240 | n02107908 | Appenzeller | 536 | 50 |
| 241 | n02108000 | EntleBucher | 338 | 50 |
| 242 | n02108089 | boxer | 1300 | 50 |
| 243 | n02108422 | bull_mastiff | 548 | 50 |
| 244 | n02108551 | Tibetan_mastiff | 859 | 50 |
| 245 | n02108915 | French_bulldog | 411 | 50 |
| 246 | n02109047 | Great_Dane | 914 | 50 |
| 247 | n02109525 | Saint_Bernard | 1037 | 50 |
| 248 | n02109961 | Eskimo_dog | 977 | 50 |
| 249 | n02110063 | malamute | 467 | 50 |
| 250 | n02110185 | Siberian_husky | 642 | 50 |
| 251 | n02110341 | dalmatian | 520 | 50 |
| 252 | n02110627 | affenpinscher | 250 | 50 |
| 253 | n02110806 | basenji | 1054 | 50 |
| 254 | n02110958 | pug | 412 | 50 |
| 255 | n02111129 | Leonberg | 487 | 50 |
| 256 | n02111277 | Newfoundland | 728 | 50 |
| 257 | n02111500 | Great_Pyrenees | 399 | 50 |
| 258 | n02111889 | Samoyed | 472 | 50 |
| 259 | n02112018 | Pomeranian | 739 | 50 |
| 260 | n02112137 | chow | 449 | 50 |
| 261 | n02112350 | keeshond | 358 | 50 |
| 262 | n02112706 | Brabancon_griffon | 486 | 50 |
| 263 | n02113023 | Pembroke | 500 | 50 |
| 264 | n02113186 | Cardigan | 518 | 50 |
| 265 | n02113624 | toy_poodle | 350 | 50 |
| 266 | n02113712 | miniature_poodle | 538 | 50 |
| 267 | n02113799 | standard_poodle | 1248 | 50 |
| 268 | n02113978 | Mexican_hairless | 353 | 50 |

### Cat / Feline

Includes domestic cats and larger wild felines under the feline branch.

- Class count: `13`
- Train image count: `7090`
- Val image count: `650`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 281 | n02123045 | tabby | 564 | 50 |
| 282 | n02123159 | tiger_cat | 473 | 50 |
| 283 | n02123394 | Persian_cat | 584 | 50 |
| 284 | n02123597 | Siamese_cat | 527 | 50 |
| 285 | n02124075 | Egyptian_cat | 591 | 50 |
| 286 | n02125311 | cougar | 566 | 50 |
| 287 | n02127052 | lynx | 510 | 50 |
| 288 | n02128385 | leopard | 529 | 50 |
| 289 | n02128757 | snow_leopard | 542 | 50 |
| 290 | n02128925 | jaguar | 524 | 50 |
| 291 | n02129165 | lion | 593 | 50 |
| 292 | n02129604 | tiger | 542 | 50 |
| 293 | n02130308 | cheetah | 545 | 50 |

### Bird

Broad bird branch covering land birds, aquatic birds, parrots, raptors, and related groups.

- Class count: `52`
- Train image count: `27942`
- Val image count: `2600`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 7 | n01514668 | cock | 584 | 50 |
| 8 | n01514859 | hen | 495 | 50 |
| 9 | n01518878 | ostrich | 512 | 50 |
| 10 | n01530575 | brambling | 430 | 50 |
| 11 | n01531178 | goldfinch | 563 | 50 |
| 12 | n01532829 | house_finch | 591 | 50 |
| 13 | n01534433 | junco | 531 | 50 |
| 14 | n01537544 | indigo_bunting | 488 | 50 |
| 15 | n01558993 | robin | 518 | 50 |
| 16 | n01560419 | bulbul | 519 | 50 |
| 17 | n01580077 | jay | 546 | 50 |
| 18 | n01582220 | magpie | 535 | 50 |
| 19 | n01592084 | chickadee | 564 | 50 |
| 20 | n01601694 | water_ouzel | 485 | 50 |
| 21 | n01608432 | kite | 441 | 50 |
| 22 | n01614925 | bald_eagle | 551 | 50 |
| 23 | n01616318 | vulture | 543 | 50 |
| 24 | n01622779 | great_grey_owl | 520 | 50 |
| 87 | n01817953 | African_grey | 576 | 50 |
| 88 | n01818515 | macaw | 558 | 50 |
| 89 | n01819313 | sulphur-crested_cockatoo | 584 | 50 |
| 90 | n01820546 | lorikeet | 614 | 50 |
| 91 | n01824575 | coucal | 510 | 50 |
| 92 | n01828970 | bee_eater | 524 | 50 |
| 93 | n01829413 | hornbill | 536 | 50 |
| 94 | n01833805 | hummingbird | 545 | 50 |
| 95 | n01843065 | jacamar | 431 | 50 |
| 96 | n01843383 | toucan | 548 | 50 |
| 97 | n01847000 | drake | 525 | 50 |
| 98 | n01855032 | red-breasted_merganser | 442 | 50 |
| 99 | n01855672 | goose | 710 | 50 |
| 100 | n01860187 | black_swan | 521 | 50 |
| 127 | n02002556 | white_stork | 517 | 50 |
| 128 | n02002724 | black_stork | 630 | 50 |
| 129 | n02006656 | spoonbill | 529 | 50 |
| 130 | n02007558 | flamingo | 488 | 50 |
| 131 | n02009229 | little_blue_heron | 542 | 50 |
| 132 | n02009912 | American_egret | 510 | 50 |
| 133 | n02011460 | bittern | 450 | 50 |
| 134 | n02012849 | crane | 591 | 50 |
| 135 | n02013706 | limpkin | 674 | 50 |
| 136 | n02017213 | European_gallinule | 489 | 50 |
| 137 | n02018207 | American_coot | 542 | 50 |
| 138 | n02018795 | bustard | 514 | 50 |
| 139 | n02025239 | ruddy_turnstone | 526 | 50 |
| 140 | n02027492 | red-backed_sandpiper | 526 | 50 |
| 141 | n02028035 | redshank | 497 | 50 |
| 142 | n02033041 | dowitcher | 641 | 50 |
| 143 | n02037110 | oystercatcher | 594 | 50 |
| 144 | n02051845 | pelican | 475 | 50 |
| 145 | n02056570 | king_penguin | 596 | 50 |
| 146 | n02058221 | albatross | 571 | 50 |

### Fish / Aquatic Vertebrate

Covers bony fish, sharks, rays, and other fish-related branches inside aquatic vertebrates.

- Class count: `16`
- Train image count: `8004`
- Val image count: `800`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 0 | n01440764 | tench | 454 | 50 |
| 1 | n01443537 | goldfish | 648 | 50 |
| 2 | n01484850 | great_white_shark | 530 | 50 |
| 3 | n01491361 | tiger_shark | 485 | 50 |
| 4 | n01494475 | hammerhead | 472 | 50 |
| 5 | n01496331 | electric_ray | 465 | 50 |
| 6 | n01498041 | stingray | 501 | 50 |
| 389 | n02514041 | barracouta | 481 | 50 |
| 390 | n02526121 | eel | 500 | 50 |
| 391 | n02536864 | coho | 541 | 50 |
| 392 | n02606052 | rock_beauty | 424 | 50 |
| 393 | n02607072 | anemone_fish | 600 | 50 |
| 394 | n02640242 | sturgeon | 453 | 50 |
| 395 | n02641379 | gar | 448 | 50 |
| 396 | n02643566 | lionfish | 509 | 50 |
| 397 | n02655020 | puffer | 493 | 50 |

### Reptile

Includes turtles, lizards, crocodilians, snakes, and dinosaur-related entries.

- Class count: `36`
- Train image count: `17819`
- Val image count: `1800`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 33 | n01664065 | loggerhead | 510 | 50 |
| 34 | n01665541 | leatherback_turtle | 498 | 50 |
| 35 | n01667114 | mud_turtle | 502 | 50 |
| 36 | n01667778 | terrapin | 502 | 50 |
| 37 | n01669191 | box_turtle | 526 | 50 |
| 38 | n01675722 | banded_gecko | 466 | 50 |
| 39 | n01677366 | common_iguana | 532 | 50 |
| 40 | n01682714 | American_chameleon | 554 | 50 |
| 41 | n01685808 | whiptail | 442 | 50 |
| 42 | n01687978 | agama | 519 | 50 |
| 43 | n01688243 | frilled_lizard | 450 | 50 |
| 44 | n01689811 | alligator_lizard | 484 | 50 |
| 45 | n01692333 | Gila_monster | 460 | 50 |
| 46 | n01693334 | green_lizard | 603 | 50 |
| 47 | n01694178 | African_chameleon | 428 | 50 |
| 48 | n01695060 | Komodo_dragon | 542 | 50 |
| 49 | n01697457 | African_crocodile | 493 | 50 |
| 50 | n01698640 | American_alligator | 554 | 50 |
| 51 | n01704323 | triceratops | 500 | 50 |
| 52 | n01728572 | thunder_snake | 450 | 50 |
| 53 | n01728920 | ringneck_snake | 468 | 50 |
| 54 | n01729322 | hognose_snake | 538 | 50 |
| 55 | n01729977 | green_snake | 526 | 50 |
| 56 | n01734418 | king_snake | 511 | 50 |
| 57 | n01735189 | garter_snake | 518 | 50 |
| 58 | n01737021 | water_snake | 483 | 50 |
| 59 | n01739381 | vine_snake | 469 | 50 |
| 60 | n01740131 | night_snake | 454 | 50 |
| 61 | n01742172 | boa_constrictor | 573 | 50 |
| 62 | n01744401 | rock_python | 428 | 50 |
| 63 | n01748264 | Indian_cobra | 500 | 50 |
| 64 | n01749939 | green_mamba | 472 | 50 |
| 65 | n01751748 | sea_snake | 497 | 50 |
| 66 | n01753488 | horned_viper | 480 | 50 |
| 67 | n01755581 | diamondback | 448 | 50 |
| 68 | n01756291 | sidewinder | 439 | 50 |

### Amphibian

Includes salamanders, newts, frogs, and related amphibian classes.

- Class count: `8`
- Train image count: `4816`
- Val image count: `400`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 25 | n01629819 | European_fire_salamander | 647 | 50 |
| 26 | n01630670 | common_newt | 536 | 50 |
| 27 | n01631663 | eft | 441 | 50 |
| 28 | n01632458 | spotted_salamander | 714 | 50 |
| 29 | n01632777 | axolotl | 540 | 50 |
| 30 | n01641577 | bullfrog | 672 | 50 |
| 31 | n01644373 | tree_frog | 683 | 50 |
| 32 | n01644900 | tailed_frog | 583 | 50 |

### Primate

Includes apes, monkeys, and lemur-adjacent primate groups.

- Class count: `20`
- Train image count: `10690`
- Val image count: `1000`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 365 | n02480495 | orangutan | 558 | 50 |
| 366 | n02480855 | gorilla | 545 | 50 |
| 367 | n02481823 | chimpanzee | 565 | 50 |
| 368 | n02483362 | gibbon | 525 | 50 |
| 369 | n02483708 | siamang | 505 | 50 |
| 370 | n02484975 | guenon | 548 | 50 |
| 371 | n02486261 | patas | 514 | 50 |
| 372 | n02486410 | baboon | 570 | 50 |
| 373 | n02487347 | macaque | 510 | 50 |
| 374 | n02488291 | langur | 585 | 50 |
| 375 | n02488702 | colobus | 564 | 50 |
| 376 | n02489166 | proboscis_monkey | 481 | 50 |
| 377 | n02490219 | marmoset | 565 | 50 |
| 378 | n02492035 | capuchin | 517 | 50 |
| 379 | n02492660 | howler_monkey | 540 | 50 |
| 380 | n02493509 | titi | 467 | 50 |
| 381 | n02493793 | spider_monkey | 499 | 50 |
| 382 | n02494079 | squirrel_monkey | 505 | 50 |
| 383 | n02497673 | Madagascar_cat | 568 | 50 |
| 384 | n02500267 | indri | 559 | 50 |

### Ungulate / Hoofed Mammal

Includes equines, swine, bovines, sheep-like, camelid, and related hoofed mammals.

- Class count: `17`
- Train image count: `9260`
- Val image count: `850`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 339 | n02389026 | sorrel | 439 | 50 |
| 340 | n02391049 | zebra | 503 | 50 |
| 341 | n02395406 | hog | 659 | 50 |
| 342 | n02396427 | wild_boar | 507 | 50 |
| 343 | n02397096 | warthog | 452 | 50 |
| 344 | n02398521 | hippopotamus | 501 | 50 |
| 345 | n02403003 | ox | 651 | 50 |
| 346 | n02408429 | water_buffalo | 493 | 50 |
| 347 | n02410509 | bison | 663 | 50 |
| 348 | n02412080 | ram | 508 | 50 |
| 349 | n02415577 | bighorn | 625 | 50 |
| 350 | n02417914 | ibex | 512 | 50 |
| 351 | n02422106 | hartebeest | 556 | 50 |
| 352 | n02422699 | impala | 451 | 50 |
| 353 | n02423022 | gazelle | 646 | 50 |
| 354 | n02437312 | Arabian_camel | 631 | 50 |
| 355 | n02437616 | llama | 463 | 50 |

### Insect

Includes beetles, flies, ants, butterflies, dragonflies, and other insect classes.

- Class count: `27`
- Train image count: `16269`
- Val image count: `1350`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 300 | n02165105 | tiger_beetle | 515 | 50 |
| 301 | n02165456 | ladybug | 717 | 50 |
| 302 | n02167151 | ground_beetle | 531 | 50 |
| 303 | n02168699 | long-horned_beetle | 541 | 50 |
| 304 | n02169497 | leaf_beetle | 624 | 50 |
| 305 | n02172182 | dung_beetle | 447 | 50 |
| 306 | n02174001 | rhinoceros_beetle | 439 | 50 |
| 307 | n02177972 | weevil | 549 | 50 |
| 308 | n02190166 | fly | 576 | 50 |
| 309 | n02206856 | bee | 557 | 50 |
| 310 | n02219486 | ant | 482 | 50 |
| 311 | n02226429 | grasshopper | 717 | 50 |
| 312 | n02229544 | cricket | 550 | 50 |
| 313 | n02231487 | walking_stick | 628 | 50 |
| 314 | n02233338 | cockroach | 562 | 50 |
| 315 | n02236044 | mantis | 729 | 50 |
| 316 | n02256656 | cicada | 664 | 50 |
| 317 | n02259212 | leafhopper | 682 | 50 |
| 318 | n02264363 | lacewing | 621 | 50 |
| 319 | n02268443 | dragonfly | 583 | 50 |
| 320 | n02268853 | damselfly | 621 | 50 |
| 321 | n02276258 | admiral | 502 | 50 |
| 322 | n02277742 | ringlet | 652 | 50 |
| 323 | n02279972 | monarch | 670 | 50 |
| 324 | n02280649 | cabbage_butterfly | 738 | 50 |
| 325 | n02281406 | sulphur_butterfly | 680 | 50 |
| 326 | n02281787 | lycaenid | 692 | 50 |

### Crustacean

Includes crabs, lobsters, crayfish, hermit crab, isopod, and related crustaceans.

- Class count: `9`
- Train image count: `4933`
- Val image count: `450`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 118 | n01978287 | Dungeness_crab | 539 | 50 |
| 119 | n01978455 | rock_crab | 597 | 50 |
| 120 | n01980166 | fiddler_crab | 492 | 50 |
| 121 | n01981276 | king_crab | 472 | 50 |
| 122 | n01983481 | American_lobster | 693 | 50 |
| 123 | n01984695 | spiny_lobster | 554 | 50 |
| 124 | n01985128 | crayfish | 491 | 50 |
| 125 | n01986214 | hermit_crab | 640 | 50 |
| 126 | n01990800 | isopod | 455 | 50 |

### Mollusk

Includes snail, slug, conch, chiton, nautilus, and related mollusk classes.

- Class count: `6`
- Train image count: `3466`
- Val image count: `300`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 112 | n01943899 | conch | 544 | 50 |
| 113 | n01944390 | snail | 618 | 50 |
| 114 | n01945685 | slug | 638 | 50 |
| 115 | n01950731 | sea_slug | 645 | 50 |
| 116 | n01955084 | chiton | 554 | 50 |
| 117 | n01968897 | chambered_nautilus | 467 | 50 |

### Vehicle

Important artifact branch covering wheeled vehicles, vessels, aircraft, spacecraft, and related conveyances.

- Class count: `67`
- Train image count: `35998`
- Val image count: `3350`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 403 | n02687172 | aircraft_carrier | 505 | 50 |
| 404 | n02690373 | airliner | 442 | 50 |
| 405 | n02692877 | airship | 509 | 50 |
| 407 | n02701002 | ambulance | 558 | 50 |
| 408 | n02704792 | amphibian | 519 | 50 |
| 417 | n02782093 | balloon | 421 | 50 |
| 428 | n02797295 | barrow | 543 | 50 |
| 436 | n02814533 | beach_wagon | 613 | 50 |
| 444 | n02835271 | bicycle-built-for-two | 495 | 50 |
| 450 | n02860847 | bobsled | 485 | 50 |
| 468 | n02930766 | cab | 485 | 50 |
| 472 | n02951358 | canoe | 552 | 50 |
| 484 | n02981792 | catamaran | 490 | 50 |
| 510 | n03095699 | container_ship | 513 | 50 |
| 511 | n03100240 | convertible | 616 | 50 |
| 537 | n03218198 | dogsled | 484 | 50 |
| 547 | n03272562 | electric_locomotive | 463 | 50 |
| 554 | n03344393 | fireboat | 509 | 50 |
| 555 | n03345487 | fire_engine | 525 | 50 |
| 561 | n03384352 | forklift | 534 | 50 |
| 565 | n03393912 | freight_car | 590 | 50 |
| 569 | n03417042 | garbage_truck | 525 | 50 |
| 573 | n03444034 | go-kart | 621 | 50 |
| 575 | n03445924 | golfcart | 535 | 50 |
| 576 | n03447447 | gondola | 652 | 50 |
| 586 | n03478589 | half_track | 472 | 50 |
| 603 | n03538406 | horse_cart | 514 | 50 |
| 609 | n03594945 | jeep | 496 | 50 |
| 612 | n03599486 | jinrikisha | 718 | 50 |
| 625 | n03662601 | lifeboat | 578 | 50 |
| 627 | n03670208 | limousine | 558 | 50 |
| 628 | n03673027 | liner | 526 | 50 |
| 656 | n03770679 | minivan | 537 | 50 |
| 657 | n03773504 | missile | 493 | 50 |
| 660 | n03776460 | mobile_home | 599 | 50 |
| 661 | n03777568 | Model_T | 503 | 50 |
| 665 | n03785016 | moped | 550 | 50 |
| 670 | n03791053 | motor_scooter | 475 | 50 |
| 671 | n03792782 | mountain_bike | 455 | 50 |
| 675 | n03796401 | moving_van | 585 | 50 |
| 690 | n03868242 | oxcart | 504 | 50 |
| 705 | n03895866 | passenger_car | 605 | 50 |
| 717 | n03930630 | pickup | 494 | 50 |
| 724 | n03947888 | pirate | 444 | 50 |
| 734 | n03977966 | police_van | 564 | 50 |
| 751 | n04037443 | racer | 492 | 50 |
| 757 | n04065272 | recreational_vehicle | 491 | 50 |
| 780 | n04147183 | schooner | 468 | 50 |
| 791 | n04204347 | shopping_cart | 513 | 50 |
| 802 | n04252077 | snowmobile | 493 | 50 |
| 803 | n04252225 | snowplow | 501 | 50 |
| 812 | n04266014 | space_shuttle | 439 | 50 |
| 814 | n04273569 | speedboat | 504 | 50 |
| 817 | n04285008 | sports_car | 608 | 50 |
| 820 | n04310018 | steam_locomotive | 520 | 50 |
| 829 | n04335435 | streetcar | 455 | 50 |
| 833 | n04347754 | submarine | 550 | 50 |
| 847 | n04389033 | tank | 551 | 50 |
| 864 | n04461696 | tow_truck | 486 | 50 |
| 866 | n04465501 | tractor | 759 | 50 |
| 867 | n04467665 | trailer_truck | 1192 | 50 |
| 870 | n04482393 | tricycle | 524 | 50 |
| 871 | n04483307 | trimaran | 494 | 50 |
| 880 | n04509417 | unicycle | 442 | 50 |
| 895 | n04552348 | warplane | 538 | 50 |
| 913 | n04606251 | wreck | 559 | 50 |
| 914 | n04612504 | yawl | 565 | 50 |

### Device

Covers instruments, machines, electronics, optical devices, musical instruments, and related device classes.

- Class count: `124`
- Train image count: `67299`
- Val image count: `6200`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 398 | n02666196 | abacus | 666 | 50 |
| 401 | n02672831 | accordion | 548 | 50 |
| 402 | n02676566 | acoustic_guitar | 449 | 50 |
| 409 | n02708093 | analog_clock | 463 | 50 |
| 413 | n02749479 | assault_rifle | 480 | 50 |
| 420 | n02787622 | banjo | 487 | 50 |
| 426 | n02794156 | barometer | 463 | 50 |
| 432 | n02804610 | bassoon | 442 | 50 |
| 447 | n02841315 | binoculars | 576 | 50 |
| 456 | n02879718 | bow | 661 | 50 |
| 464 | n02910353 | buckle | 499 | 50 |
| 470 | n02948072 | candle | 682 | 50 |
| 471 | n02950826 | cannon | 618 | 50 |
| 475 | n02965783 | car_mirror | 547 | 50 |
| 476 | n02966193 | carousel | 465 | 50 |
| 479 | n02974003 | car_wheel | 631 | 50 |
| 480 | n02977058 | cash_machine | 577 | 50 |
| 486 | n02992211 | cello | 472 | 50 |
| 491 | n03000684 | chain_saw | 539 | 50 |
| 494 | n03017168 | chime | 547 | 50 |
| 507 | n03075370 | combination_lock | 524 | 50 |
| 508 | n03085013 | computer_keyboard | 679 | 50 |
| 513 | n03110669 | cornet | 506 | 50 |
| 517 | n03126707 | crane | 649 | 50 |
| 527 | n03180011 | desktop_computer | 516 | 50 |
| 530 | n03196217 | digital_clock | 455 | 50 |
| 531 | n03197337 | digital_watch | 463 | 50 |
| 535 | n03208938 | disk_brake | 435 | 50 |
| 541 | n03249569 | drum | 455 | 50 |
| 545 | n03271574 | electric_fan | 509 | 50 |
| 546 | n03272010 | electric_guitar | 439 | 50 |
| 558 | n03372029 | flute | 533 | 50 |
| 566 | n03394916 | French_horn | 495 | 50 |
| 571 | n03425413 | gas_pump | 534 | 50 |
| 577 | n03447721 | gong | 532 | 50 |
| 579 | n03452741 | grand_piano | 500 | 50 |
| 583 | n03467068 | guillotine | 535 | 50 |
| 584 | n03476684 | hair_slide | 579 | 50 |
| 589 | n03483316 | hand_blower | 552 | 50 |
| 590 | n03485407 | hand-held_computer | 440 | 50 |
| 592 | n03492542 | hard_disc | 545 | 50 |
| 593 | n03494278 | harmonica | 434 | 50 |
| 594 | n03495258 | harp | 561 | 50 |
| 595 | n03496892 | harvester | 547 | 50 |
| 600 | n03532672 | hook | 640 | 50 |
| 604 | n03544143 | hourglass | 564 | 50 |
| 607 | n03590841 | jack-o'-lantern | 427 | 50 |
| 613 | n03602883 | joystick | 470 | 50 |
| 616 | n03627232 | knot | 443 | 50 |
| 620 | n03642806 | laptop | 523 | 50 |
| 626 | n03666591 | lighter | 515 | 50 |
| 632 | n03691459 | loudspeaker | 416 | 50 |
| 633 | n03692522 | loupe | 571 | 50 |
| 635 | n03706229 | magnetic_compass | 578 | 50 |
| 641 | n03720891 | maraca | 497 | 50 |
| 642 | n03721384 | marimba | 474 | 50 |
| 645 | n03733131 | maypole | 674 | 50 |
| 650 | n03759954 | microphone | 545 | 50 |
| 673 | n03793489 | mouse | 495 | 50 |
| 674 | n03794056 | mousetrap | 498 | 50 |
| 676 | n03803284 | muzzle | 611 | 50 |
| 677 | n03804744 | nail | 563 | 50 |
| 678 | n03814639 | neck_brace | 629 | 50 |
| 681 | n03832673 | notebook | 458 | 50 |
| 683 | n03838899 | oboe | 565 | 50 |
| 684 | n03840681 | ocarina | 435 | 50 |
| 685 | n03841143 | odometer | 535 | 50 |
| 686 | n03843555 | oil_filter | 491 | 50 |
| 687 | n03854065 | organ | 667 | 50 |
| 691 | n03868863 | oxygen_mask | 642 | 50 |
| 694 | n03874293 | paddlewheel | 686 | 50 |
| 695 | n03874599 | padlock | 780 | 50 |
| 699 | n03884397 | panpipe | 416 | 50 |
| 704 | n03891332 | parking_meter | 591 | 50 |
| 714 | n03929660 | pick | 443 | 50 |
| 718 | n03933933 | pier | 469 | 50 |
| 723 | n03944341 | pinwheel | 517 | 50 |
| 739 | n03992509 | potter's_wheel | 618 | 50 |
| 740 | n03995372 | power_drill | 421 | 50 |
| 744 | n04008634 | projectile | 734 | 50 |
| 745 | n04009552 | projector | 559 | 50 |
| 753 | n04040759 | radiator | 450 | 50 |
| 755 | n04044716 | radio_telescope | 548 | 50 |
| 758 | n04067472 | reel | 665 | 50 |
| 761 | n04074963 | remote_control | 617 | 50 |
| 763 | n04086273 | revolver | 426 | 50 |
| 764 | n04090263 | rifle | 486 | 50 |
| 769 | n04118776 | rule | 501 | 50 |
| 772 | n04127249 | safety_pin | 522 | 50 |
| 776 | n04141076 | sax | 508 | 50 |
| 778 | n04141975 | scale | 507 | 50 |
| 782 | n04152593 | screen | 603 | 50 |
| 783 | n04153751 | screw | 494 | 50 |
| 795 | n04228054 | ski | 488 | 50 |
| 798 | n04238763 | slide_rule | 489 | 50 |
| 800 | n04243546 | slot | 747 | 50 |
| 801 | n04251144 | snorkel | 556 | 50 |
| 807 | n04258138 | solar_dish | 622 | 50 |
| 811 | n04265275 | space_heater | 605 | 50 |
| 815 | n04275548 | spider_web | 688 | 50 |
| 818 | n04286575 | spotlight | 551 | 50 |
| 822 | n04311174 | steel_drum | 459 | 50 |
| 823 | n04317175 | stethoscope | 549 | 50 |
| 826 | n04328186 | stopwatch | 625 | 50 |
| 827 | n04330267 | stove | 1196 | 50 |
| 828 | n04332243 | strainer | 469 | 50 |
| 835 | n04355338 | sundial | 528 | 50 |
| 836 | n04355933 | sunglass | 522 | 50 |
| 837 | n04356056 | sunglasses | 606 | 50 |
| 843 | n04371774 | swing | 492 | 50 |
| 844 | n04372370 | switch | 495 | 50 |
| 845 | n04376876 | syringe | 575 | 50 |
| 856 | n04428191 | thresher | 634 | 50 |
| 862 | n04456115 | torch | 687 | 50 |
| 872 | n04485082 | tripod | 472 | 50 |
| 875 | n04487394 | trombone | 483 | 50 |
| 878 | n04505470 | typewriter_keyboard | 523 | 50 |
| 881 | n04515003 | upright | 421 | 50 |
| 886 | n04525305 | vending_machine | 541 | 50 |
| 889 | n04536866 | violin | 513 | 50 |
| 892 | n04548280 | wall_clock | 480 | 50 |
| 902 | n04579432 | whistle | 510 | 50 |
| 908 | n04592741 | wing | 489 | 50 |
| 916 | n06359193 | web_site | 568 | 50 |

### Container

Includes bags, baskets, bottles, jars, vessels, tanks, and other container-like artifacts.

- Class count: `48`
- Train image count: `25259`
- Val image count: `2400`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 412 | n02747177 | ashcan | 503 | 50 |
| 414 | n02769748 | backpack | 618 | 50 |
| 427 | n02795169 | barrel | 565 | 50 |
| 435 | n02808440 | bathtub | 661 | 50 |
| 438 | n02815834 | beaker | 650 | 50 |
| 440 | n02823428 | beer_bottle | 603 | 50 |
| 441 | n02823750 | beer_glass | 582 | 50 |
| 463 | n02909870 | bucket | 650 | 50 |
| 478 | n02971356 | carton | 436 | 50 |
| 481 | n02978881 | cassette | 542 | 50 |
| 492 | n03014705 | chest | 586 | 50 |
| 503 | n03062245 | cocktail_shaker | 422 | 50 |
| 504 | n03063599 | coffee_mug | 487 | 50 |
| 519 | n03127925 | crate | 578 | 50 |
| 549 | n03291819 | envelope | 486 | 50 |
| 572 | n03443371 | goblet | 440 | 50 |
| 588 | n03482405 | hamper | 505 | 50 |
| 618 | n03633091 | ladle | 479 | 50 |
| 636 | n03709823 | mailbag | 433 | 50 |
| 637 | n03710193 | mailbox | 492 | 50 |
| 647 | n03733805 | measuring_cup | 503 | 50 |
| 653 | n03764736 | milk_can | 510 | 50 |
| 666 | n03786901 | mortar | 601 | 50 |
| 692 | n03871628 | packet | 599 | 50 |
| 709 | n03908618 | pencil_box | 452 | 50 |
| 719 | n03935335 | piggy_bank | 443 | 50 |
| 720 | n03937543 | pill_bottle | 595 | 50 |
| 725 | n03950228 | pitcher | 580 | 50 |
| 728 | n03958227 | plastic_bag | 467 | 50 |
| 737 | n03983396 | pop_bottle | 563 | 50 |
| 738 | n03991062 | pot | 445 | 50 |
| 748 | n04026417 | purse | 522 | 50 |
| 756 | n04049303 | rain_barrel | 572 | 50 |
| 771 | n04125021 | safe | 467 | 50 |
| 773 | n04131690 | saltshaker | 521 | 50 |
| 790 | n04204238 | shopping_basket | 481 | 50 |
| 797 | n04235860 | sleeping_bag | 486 | 50 |
| 804 | n04254120 | soap_dispenser | 450 | 50 |
| 868 | n04476259 | tray | 426 | 50 |
| 876 | n04493381 | tub | 525 | 50 |
| 883 | n04522168 | vase | 541 | 50 |
| 893 | n04548362 | wallet | 497 | 50 |
| 896 | n04553703 | washbasin | 596 | 50 |
| 898 | n04557648 | water_bottle | 467 | 50 |
| 899 | n04560804 | water_jug | 610 | 50 |
| 900 | n04562935 | water_tower | 661 | 50 |
| 901 | n04579145 | whiskey_jug | 459 | 50 |
| 907 | n04591713 | wine_bottle | 502 | 50 |

### Clothing

Includes garments, outerwear, hats, footwear, neckwear, and other clothing-related classes.

- Class count: `48`
- Train image count: `25918`
- Val image count: `2400`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 399 | n02667093 | abaya | 445 | 50 |
| 400 | n02669723 | academic_gown | 571 | 50 |
| 411 | n02730930 | apron | 613 | 50 |
| 433 | n02807133 | bathing_cap | 485 | 50 |
| 439 | n02817516 | bearskin | 446 | 50 |
| 445 | n02837789 | bikini | 581 | 50 |
| 451 | n02865351 | bolo_tie | 501 | 50 |
| 452 | n02869837 | bonnet | 613 | 50 |
| 457 | n02883205 | bow_tie | 589 | 50 |
| 459 | n02892767 | brassiere | 487 | 50 |
| 465 | n02916936 | bulletproof_vest | 431 | 50 |
| 474 | n02963159 | cardigan | 665 | 50 |
| 496 | n03026506 | Christmas_stocking | 582 | 50 |
| 515 | n03124170 | cowboy_hat | 529 | 50 |
| 518 | n03127747 | crash_helmet | 588 | 50 |
| 529 | n03188531 | diaper | 539 | 50 |
| 552 | n03325584 | feather_boa | 535 | 50 |
| 560 | n03379051 | football_helmet | 509 | 50 |
| 568 | n03404251 | fur_coat | 581 | 50 |
| 578 | n03450230 | gown | 489 | 50 |
| 601 | n03534580 | hoopskirt | 461 | 50 |
| 608 | n03594734 | jean | 447 | 50 |
| 610 | n03595614 | jersey | 463 | 50 |
| 614 | n03617480 | kimono | 555 | 50 |
| 615 | n03623198 | knee_pad | 453 | 50 |
| 617 | n03630383 | lab_coat | 524 | 50 |
| 638 | n03710637 | maillot | 463 | 50 |
| 639 | n03710721 | maillot | 531 | 50 |
| 652 | n03763968 | military_uniform | 622 | 50 |
| 655 | n03770439 | miniskirt | 632 | 50 |
| 658 | n03775071 | mitten | 508 | 50 |
| 667 | n03787032 | mortarboard | 501 | 50 |
| 689 | n03866082 | overskirt | 417 | 50 |
| 697 | n03877472 | pajama | 554 | 50 |
| 735 | n03980874 | poncho | 624 | 50 |
| 775 | n04136333 | sarong | 517 | 50 |
| 785 | n04162706 | seat_belt | 549 | 50 |
| 793 | n04209133 | shower_cap | 853 | 50 |
| 806 | n04254777 | sock | 684 | 50 |
| 808 | n04259630 | sombrero | 583 | 50 |
| 824 | n04325704 | stole | 599 | 50 |
| 834 | n04350905 | suit | 437 | 50 |
| 841 | n04370456 | sweatshirt | 467 | 50 |
| 842 | n04371430 | swimming_trunks | 594 | 50 |
| 869 | n04479046 | trench_coat | 514 | 50 |
| 887 | n04532106 | vestment | 587 | 50 |
| 903 | n04584207 | wig | 497 | 50 |
| 906 | n04591157 | Windsor_tie | 503 | 50 |

### Food Fruit

Fruit-like food branch under edible fruit.

- Class count: `10`
- Train image count: `5334`
- Val image count: `500`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 948 | n07742313 | Granny_Smith | 515 | 50 |
| 949 | n07745940 | strawberry | 474 | 50 |
| 950 | n07747607 | orange | 621 | 50 |
| 951 | n07749582 | lemon | 562 | 50 |
| 952 | n07753113 | fig | 461 | 50 |
| 953 | n07753275 | pineapple | 520 | 50 |
| 954 | n07753592 | banana | 560 | 50 |
| 955 | n07754684 | jackfruit | 501 | 50 |
| 956 | n07760859 | custard_apple | 466 | 50 |
| 957 | n07768694 | pomegranate | 654 | 50 |

### Prepared Food / Dish

Prepared foods and dish-like classes such as soup, sandwich, pizza, burrito, and related foods.

- Class count: `11`
- Train image count: `6172`
- Val image count: `550`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 659 | n03775546 | mixing_bowl | 452 | 50 |
| 712 | n03920288 | Petri_dish | 461 | 50 |
| 809 | n04263257 | soup_bowl | 559 | 50 |
| 925 | n07584110 | consomme | 649 | 50 |
| 926 | n07590611 | hot_pot | 561 | 50 |
| 933 | n07697313 | cheeseburger | 479 | 50 |
| 934 | n07697537 | hotdog | 533 | 50 |
| 962 | n07871810 | meat_loaf | 627 | 50 |
| 963 | n07873807 | pizza | 665 | 50 |
| 964 | n07875152 | potpie | 678 | 50 |
| 965 | n07880968 | burrito | 508 | 50 |

### Beverage

Drink categories such as wine, coffee, punch-like drinks, and related beverage classes.

- Class count: `4`
- Train image count: `2149`
- Val image count: `200`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 966 | n07892512 | red_wine | 493 | 50 |
| 967 | n07920052 | espresso | 617 | 50 |
| 968 | n07930864 | cup | 511 | 50 |
| 969 | n07932039 | eggnog | 528 | 50 |

### Person

Direct person-related classes under the WordNet `person` branch.

- Class count: `3`
- Train image count: `1625`
- Val image count: `150`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 981 | n09835506 | ballplayer | 569 | 50 |
| 982 | n10148035 | groom | 583 | 50 |
| 983 | n10565667 | scuba_diver | 473 | 50 |

### Fungus

Mushroom and fungus-related biological classes.

- Class count: `7`
- Train image count: `3656`
- Val image count: `350`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 991 | n12985857 | coral_fungus | 493 | 50 |
| 992 | n12998815 | agaric | 623 | 50 |
| 993 | n13037406 | gyromitra | 485 | 50 |
| 994 | n13040303 | stinkhorn | 560 | 50 |
| 995 | n13044778 | earthstar | 457 | 50 |
| 996 | n13052670 | hen-of-the-woods | 443 | 50 |
| 997 | n13054560 | bolete | 595 | 50 |

### Geological Formation

Natural scene / landform branch including cliffs, shores, reefs, valleys, and mountains.

- Class count: `10`
- Train image count: `7165`
- Val image count: `500`

| Index | WNID | Class Name | Train Count | Val Count |
| --- | --- | --- | --- | --- |
| 970 | n09193705 | alp | 573 | 50 |
| 972 | n09246464 | cliff | 721 | 50 |
| 973 | n09256479 | coral_reef | 602 | 50 |
| 974 | n09288635 | geyser | 791 | 50 |
| 975 | n09332890 | lakeside | 753 | 50 |
| 976 | n09399592 | promontory | 557 | 50 |
| 977 | n09421951 | sandbar | 702 | 50 |
| 978 | n09428293 | seashore | 1268 | 50 |
| 979 | n09468604 | valley | 730 | 50 |
| 980 | n09472597 | volcano | 468 | 50 |
