# Acceptance Criterion: Human Factors and Clinical Use

Last edited time: March 22, 2023 10:00 PM
Owner: Anonymous

**Table of Contents**

# 1. Baseline MVP Requirements

Before we begin: Portions of the initial part of this guidance are also captured in the engineering specification document, but a quick reminder of expectations before human factors is warranted on a project of this complexity. In this document "the Device" refers to the assembled headset-visor combination as well as the U/X elements and functionality the donner will be exposed to. It is assumed we have no control over the tooling or other characteristics of the smartphone, so that is not considered part of the Device except insomuch as it impacts overall weight and heat.

The purpose of this investigation - and of sharing it with internal and external teams in this way - is to provide medical guidance informed by practice and research to better shape our eventual prototype. In the course of this work I evaluated 167 different potential addressable conditions to narrow down candidates classified as ideal, likely yes, likely no, and excluded from our trials for various reasons. A table with comprehensive clinical details can be found [here](https://www.notion.so/8dfb94d7f8f64ecabeb06fd98c5e6a33) for the curious, and a more abbreviated and practical version [here](https://www.notion.so/0a46504db4b4493a85a1df70cde4628f) for functional reference. 

As a quick preview, however: From both humanitarian and market-facing standpoints, the results are very good news.

- First, to frame the data: the population numbers and recommendations here are restricted to demographic data from the United States. While I have extrapolated this data to twenty or so other countries with mature funding systems, I think I'll take a break from articulating all that until v2 if we start offering other languages.
- Out of 167 conditions, 25 were rated as "Ideal" or "Likely Yes," which may seem like a small percentage but fortunately constitutes the bulk of the addressable population. More frequently excluded are so-called "low-incidence" disorders, which are uncommon for a reason - rarity and high mortality.
- Details around the research and the math behind incidence and prevalence and my methodology generally is covered in the [clinical documentation](https://www.notion.so/8dfb94d7f8f64ecabeb06fd98c5e6a33), but suffice to say that this was an exhaustive investigation and I have high confidence in the numbers I will share.

## 1.2 Broad Requirements for Use by Target Cohort(s)

As the MVP/prototype of the wearable is generally intended for experimental and research audiences, the use requirements are marginally more broad than what would be expected for special populations only. As a result, as will be detailed later, certain population cohorts have been excluded from targeting for this iteration. However, the following statements of inclusion hold true.

- Positioning:
    - Must be usable in a seated position.
    - Must be usable in a reclining position with weight on the back of the head for a minimum of two hours.
    - Must be usable while in motion.
    - Must be usual by a typical user sitting upright for a minimum of four hours without discomfort, ideally six.
    - Must be easily removed when the user's scalp needs cleaned or wiped.

- Environment:
    - Must be usable in clinical and laboratory settings.
    - Must have a display bright enough to overcome ambient light.
    - Must be usable in home and other environments of daily living without engineering or clinical supervision.

- Functional Use Characteristics:
    - Must be able to fit prescription glasses beneath the visor and still function appropriately.
    - Must be adjustable to accommodate the range of sizes of an adult human head.
    - Must allow for adequate cooling egress and not cause discomfort due to heat.
    - Adjustments and replacement of removable parts must be accomplishable by a non-technical person with minimal tools.
    - Visor must allow eye contact between user and interlocutor.
    - Caregiver should be able to evaluate EEG strength and device power at a glance.
    - Onboard audio and haptic driver must be adjustable by user and caregiver.
    - Power on/off and wand movements should be accessible to the user.
    - Industrial design should be stylish, appealing, and avoid a "medical" appearance.

- Special Accommodations:
    - Must be able to remain intact in event of chorea (involuntary muscle movement).
    - Must be light enough or suspended in a fashion that accounts for low muscle tone in the head, neck, and trunk.
    - Must consider the possibility of sensitivity to ambient light, contrast needs due to cortical visual impairment (CVI) and similar, and other sensory concerns.
    - Must consider the fact that different target populations will have different visual and interactive needs (e.g. Cerebral Palsy vs Aphasia).
    - Must account for - or accept exclusion of - comorbid dermal conditions, including allergies.
    - Minimally, BCI calibration must be usable by individuals with severely compromised vision. U/X adjustment to accommodate for such needs should be considered.

## 1.3 Non-Requirements for MVP

- Functional Use Characteristics:
    - A waterproof rating allowing for immersion is not expected (but welcome).
    - Only typically developed adult head sizes are expected to be supported.
    - All allergic and autoimmune responses cannot be accounted for.
    - We can assume a caregiver will be involved in setup and removal of the device.

# 2. Human Factors: Clinical Evaluation

The primary purpose of this resource and all of the literature reviewed in its creation was to determine the "addressable" population of this wearable - not just the *ideal,* or the *complete unlimited total*, but those who would be well-served by the experience. **In all things our maxim is Do No Harm**. For that reason there are certain individuals that *may* benefit, but our trial team and partners are not being instructed to pursue that diagnostic population as part of the target cohort. For granular detail on those diagnoses and clinical reasoning, see the clinical reference tables [here](https://www.notion.so/8dfb94d7f8f64ecabeb06fd98c5e6a33), and for specific case studies demonstrating benefit see the personas detailed [here](https://www.notion.so/0c3ad1ce57124cc9adbe90d62650e0f0).

There are some populations that clearly and regularly will be included as AR/BCI wearable candidates. Others may be good candidates for one and not the other, thus the requirement of separability. Others are clearly excluded in either category for this iteration. As a general set of guidance, however, before specifics:

- Even in cases like locked-in syndrome, where the need (and its cause) are very clear, a medical professional should be involved in the trial; it's possible, for example, that the individual has photosensitive epilepsy but that fact is not apparent because they are nonverbal and paralyzed.
- Even if a condition is documented as *not* having (again, as an example) a comorbid allergy to silver or latex or any other hypothetical electrical element, it is essential that an attending nurse or doctor review their record to verify. Judgments about factors like that are not for this team or even pathologists to determine on the fly.
- The user's comfort takes priority over any engineering or clinical judgment. If there is no apparent reason why they say they wish to remove the device, but they request it, we honor that decision. It has the potential to become their voice and, in that moment, it is theirs.

- Initial environments should be chosen that have adequate color and lighting contrast to ensure visibility on the display and no confusion of colors or textures between the physical space and U/X elements.
- Initial trials should occur in comfortable, preferably seated environments, with adults present in case of disorientation that could lead to dizziness or a fall.
- In the event that there is an adverse reaction, a licensed counselor or similar professional should be engaged to assist the user through their experience. This is also true prior even to initiation of a trial when a user or their family/caregivers express psychological concerns.

## 2.1 Human Factors: Exclusion

With those caveats aside, the following are disqualifying symptoms and diagnoses for which a trial is **expressly excluded**. **No trial should occur when conditions from 2.1.1-2.1.4 are involved without express permission from the Cognixion clinical team and attending medical professionals, even if you or the user feels otherwise.**

### 2.1.1 Exclusion due to Potential Harm

**Photosensitive Epilepsy**

- Also known as Photoparoxysmal Response Epilepsy; as this was a primary risk entering this project and one with potentially fatal consequences, it is one we have taken very seriously. A thorough examination ("deep dive") of the subject can be found [here](https://www.notion.so/4eaf6818666b498488151ef4544127de). In short, the frequency at which steady state evoked potentials are most effective - our calibration mechanism - are precisely those most likely to cause a seizure in a photosensitive individual (~5-30fps).
- While this form of epilepsy is very uncommon (~3% of all seizure forms) and tends to resolve in adulthood, any history of this condition is disqualifying and potentially highly dangerous. The only conceivable exception would occur with the express consent and probable presence of specialized neurology and optometry staff.
- Due to the risk involved, potential users should be screened for past seizure incidents and if they report positively, even in the absence of known photosensitivity, a physician should be consulted to evaluate and consent before a trial begins.

**Albinism**

- While photosensitive-type epilepsy constitutes only 3% of general seizure activity, a genetic relationship among individuals who are albino has repeatedly shown that their incidence of photosensitivity approaches 80-90%. Albinism is therefore an automatic disqualifying characteristic.
- Retinal shape and ocular nerve development are also impacted in albinism (in two variants - "oculotaneous" and "ocular"), so epilepsy is not the only factor.

**Skin Conditions Related to Specific Disorders**

- Because the research into these populations led to so-called "low-incidence" groups, often surprising comorbidities (literally meaning "living with" - secondary conditions resulting from the primary) have to be discussed. One such example is **aplasia cutis congenita**, which causes skin simply not to grow over muscle and bone across portions of the body. This is clearly disqualifying, and is associated with a number of conditions we would otherwise address, most notably Trisomy 13 (Patau syndrome). A similar condition is **liodystraphy**, which is the failure to create fatty tissues, resulting in extremely sensitive and delicate skin and therefore disqualifying CANDLE Syndrome from our target cohorts.
- Conversely, some conditions - such as Cardiofaciocutaneous (CFC) syndrome - present with **icthyosis**, which is extremely dry, thick, scaly skin across their entire bodies. No literature exists of research into electrode efficacy in that environment but my clinical assumption is that it would be potentially compromised.
- Another potential risk symptom is **nummular dermatitis**, which is a form of eczema resulting lesions and easily torn skin. Unfortunately, it is a side effect (thankfully uncommon) of many medications taken to reduce involuntary muscle movement (chorea) and advanced diabetes. That, combined with age, makes fragility of skin - and discomfort at prolonged touch - more common than average in conditions like Parkinsons. This is not a disqualifying factor but one warranting family and medical attention; there are topical solutions to address it, but if unnoticed can cause injury.
- Of particular concern, and fortunately also particularly infrequent in the United States, is an infection, infestation, or **neoplasm** of the scalp. This condition is caused by bacteria, fungus, and protozoa who nest in human hair and dissolve skin tissue. Left untreated, cysts and even fatal infection can result; obviously those conditions would be exacerbated by any sustained cranial dermal contact. Fortunately, these forms of folliculitus, keloidalis, and cellulitis can be treated with antibiotics and are typically noticed by the affected individual long before critical stage, but any sign of abnormal skin coloring or strength in the scalp should result in a halt of the trial so a dermatologist can evaluate and consent.
- Finally, the combination of age, sun exposure, smoking, and even prescription steroid use can cause thinning of the skin in geriatric populations. This is again not an inherently disqualifying factor, but underscores the gentleness with which fitting should be approached and the care that must be paid to design for a device that may spend extended periods in contact with the rear of the head while reclining.

**Allergies, Contact Dermatitis, and Anaphlaxis**

- It is extremely unlikely that skin contact with any material used in the device or electrodes would result in anaphlaxis, a potentially fatal allergic reaction. However, such dermatitis can be very painful and persistent, and autoimmune disorders resulting in unusual allergies frequently co-occur with many of the cohorts we are targeting. Particularly problematic may in fact be cleaners and creams that hypothetically could be used in the course of electrode positioning, but allergies to other materials occur as well.
- The most common allergic reactions likely to be encountered in trials are those to rubber/latex, sanitizing agents, and - depending on materials - certain epoxy resins used in 3D printing.
- Allergies to metals and alloys are also common, although typically less severe. Gold is likely to cause reactions in the largest number of people, particularly when alloyed with nickel. Palladium, cobalt, mercury, and even aluminum can also cause adverse effects, particularly with prolonged exposure. Stainless steel and titanium are relatively risk-free, as is pure silver (research shows that those who report reactions to silver have almost always actually been exposed to a nickel alloy).
- As dentists, jewelers, and countless other professionals have determined, it is likely impossible that we can account for every possible dermal allergic reaction. Some individuals may have autoimmune comorbidities that exclude them from use of this device - and if the severity is such that they cannot use the headset, it's likely that they will already be aware of their sensitivity. However, when conducting trials clinicians and observers should monitor for skin that grows red, begins to itch or swell, forms bumps or blisters, or grows hot to the touch, at which point the trial should be discontinued pending an allergy test.
- Like other comorbidities above, this potential for harm underscores the need for a medical history review by a qualified professional before trials are attempted. One practical recommendation would be to have multiple options available in terms of electrode composition and any dermal treating agents, but while that's an excellent long-term goal it is not an MVP requirement and may constitute a costly sideline to solve a rare problem.

### 2.1.2 Exclusion due to Sensory Inadequacy

**Hearing Loss**

- Worth noting as a sideline to be aware of in trials is the fact that any condition characterized by craniofacial abnormality - an unusual looking face or skull - is typically accompanied by hearing loss. Down Syndrome is the hallmark example of this, as the population almost without exception lacks functioning eustachian tubes, thus feeling like they're "underwater" with average 70% lower hearing acuity than the general population. This is by no means disqualifying, but something to be sensitive to in trials and worth considering in U/X to pair visual and/or haptic elements with any auditory cues.
- Bluetooth prototcol connection to existing hearings - protocols need more exploration (@norm@cognixion.com @wil macaulay @Cris Micheli ).
- Cochlear Implants - Tenative research and clinical trials shows this population may need to be excluded, but further information is warranted.

**Visual Impairment**

- Calibration for BCI, and interaction with AR, do however require at least some visual functioning - although the extent remains somewhat to be determined. We do know that even individuals with severely compromised vision can successfully calibrate, albeit with higher latency and overall investment of time. As many of our target conditions have visual comorbidities ranging from jittering ocular muscles (nystagmus) to optical nerve atrophy, we should prepare for and anticipate challenges in that regard, particularly among individuals who also experience cognitive or behavioral challenges that impact patience or understanding.
- Among the conditions that *may* be disqualified due to ocular abnormalities are: Trisomy 9; Mowat-Wilson Syndrome; Guillian-Barré syndrome, CHARGE syndrome, Joubert syndrome, Machado-Joseph disease, and Cornelia de Lange syndrome. All are low-incidence and have a high rate of visual disorders but not outright blindness - although we should also remember that visual disorders can occur in *anyone;* genetic relationships simply make them more predictable in cases like these.
- However, there are some conditions that *absolutely* manifest in blindness, either at birth or progressively, and are therefore excluded. Among them are MERRF syndrome (diagnosed individuals also have photosensitive epilepsy before onset of blindness, so that's a double no), Leigh syndrome, Dandy-Walker malformation, and Batten disease (specifically the CLN2 variant). Clinical recruiting teams have detailed information to assist them in ruling out these candidates.

### 2.1.3 Exclusion due to Physical Characteristics

**Cranofacial Abnormalities and Morphology**

- It has already been established that our wearable MVP will address an adult population due to the design and production complications inherent in also sizing for children; thus, pediatric populations have been ruled out - particularly those with average mortality before the end of adolescence. It is worth noting, however, that an AR solution without BCI could conceivably be fitted, and candidacy research has been conducted on juvenile conditions if we wish to pursue a headband-with-lens variant, although the U/X of such a solution would have to address considerably different needs.
- Many if not most of the conditions addressable by this solution involve either the *deletion* of critical genetic material (e.g. Rett syndrome), the *addition* of genetic material (e.g. Huntingtons and most other progressive dementias), or the wholesale *duplication* of entire chromosomes (e.g. Down syndrome). Of these groups, it is almost invariably the third - chromosomal disorders - that present not only with cognitive, motor, and speech and language conditions but also craniofacial and orthopedic abnormalities related to the development of their bone structure.
- Abnormal cranial morphology (shape) is not in itself disqualifying, although trials will reveal the extent to which electrode placement is compromised, particularly when EEG activity seems similarly disrupted/rerouted as in Angelman syndrome. However, certain conditions accompanied by micro- (underdeveloped) or macrocephaly (overdeveloped cranial structure) can be preemptively considered unlikely candidates.
- Among the "unlikely but possible" are conditions that, for the most part, present with microcephaly but otherwise are likely to mature to adulthood. Examples include Christianson syndrome, Trisomy 12p, Trisomy 9, Mowat-Wilson syndrome, Guillain-Barré Syndrome (famously the result of the Zika virus, but also found recently among live births infected with the novel coronavirus), CHARGE syndrome, Patau syndrome, Aicardi syndrome, DiGeorge syndrome, Pitt-Hopkins syndrome, Prader-Willi syndrome (easy to diagnose due to presence of webbed feet and tail), Cornelia de Lange syndrome, and Neonatal Encephalopathy (historically "water on the brain"). For any of the conditions above I would cautiously proceed if not excluded for other reasons but expect substantial adjustments in fitting.
- There are no conditions outright excluded due to micro- or macrocephaly but that may change pending human factors trials, either via determination of non-candidacy or presentation of an AR-only solution.

**Electrode Occlusion or Otherwise Compromised Occipital Lobe Placement**

- For reasons detailed in the more comprehensive clinical document here, brain injuries acquired from outside trauma were excluded from consideration in this work even though they very frequently could pose strong candidacy. However, it's worth noting - obvious though it may be - that cranial damage and resulting brain injury that (for example) removes the occipital lobe entirely would be disqualifying.
- It is very rare, but conditions like Dandy-Walker malformation present with **hypertrichosis**, which is extremely excessive hair growth on the head, neck, and much of the body. It remains to be seen whether this is an obstacle, but there's no reason not to proceed with a trial if such an individual is encountered.

### 2.1.4 Exclusion due to Capacity or Efficacy

**Psychosis and Other Cognitive-Perceptual Conditions**

- Due to the nature of the equipment - both in terms of value and precision in positioning - disorders that either present concurrently with psychosis or violent behavior or progress into similar have been preemptively excluded. Even if we could validate that the user understood and was benefiting from the AR, for example, it's extremely likely it would intentionally or inadvertently be removed. Some examples include Heller Syndrome ("childhood disintegrative disorder"), PPM-X (Lindsay-Burn syndrome), and DiGeorge Syndrome (22q.11 deletion).
- It's worth emphasizing that the above comment may be an overgeneralization, and exceptions likely exist; also, there is no value judgment ascribed to the word "psychosis" in this context, simply a statement of cognitive-perceptual fact. In contrast, very similar dysfunction can be seen in late-stage dementias such as Alzheimer's and Huntingtons, but I have *not* excluded those from candidacy because it is assumed that by the time they reach an equally agitated state they will be physically compromised to an extent that prevents them from damaging or removing the device.

**Intellectual Disability**

- In clinical practice, an overriding ethical mandate is the presumption of competence - e.g., the responsibility to assume from the outset that everyone is capable of using augmentative/alternative communication. For that reason I make no exclusions on the basis of cognitive functioning, but from years of practice and research I can attest that there is in fact an underlying need to understand - or be able to be taught - fundamental cause and effect relationships in order to effectively communicate. Unfortunately, I have had patients in the past - after a traumatic brain injury, or due to hydrocephalus, or in the case of Prader-Willi and other severely impactful disorders - where I have been unable to elicit any sort of intentional response.
- Again, we should not deny a trial on this basis, and must always assume the capacity for growth, so this is perhaps wrongly labeled as an exclusion criterion. But it is equally important that the team, caregivers, and any involved professionals not view this wearable as a magic wand that will spontaneously resolve a profound disability. It is a tool - the user is the hero of this journey, not our headset - and in the end, help as we may, it is up to them whether to take it up. Set expectations appropriately - underpromise and overdeliver, especially with v1 - a maxim best adopted everywhere from the room during a trial to marketing and advertising and the way we describe our work to friends.

### 2.1.5 Non-Excluding Factors that Warrant Caution

**Content and Experience**

- Among the potentially included target cohorts are individuals with vastly different needs not just in terms of their condition and level of functioning but also in terms of their environmental context, daily routines, preferred use of language, familiarity with technology, and much more. This has considerable implications for U/X, where a user with Aphasia - as demonstrably and repeatedly reported by research - responds to completely different visual stimuli as, say, a 20 year old girl with Rett syndrome. If we wish to address all of these populations at a high level, we need U/X variants that account for that - not to mention training, support, and the out of the box experience.
- Visual occlusion due to AR elements in a device inherently designed for functional use while mobile could create situations where users are more likely to be threatened by physical hazards. U/X must provide enough information to be functional while also allowing for adequate environmental awareness.

**Additional Physical Factor Concerns**

- A portion of the potential thermal hazard is out of our control, in the form of the paired smartphone projecting the visor display. As this population is fragile by nature, we have a high bar in terms of the challenge posed by ventilation - not to mention the expectations users will have around water resistance and inclement weather, which are currently met reasonably well in the tablet-based AAC industry (~IP67).
- A number of the conditions we are targeting routinely receive cerebellar shunts to relieve cranial pressure - shunts that are known to migrate and be otherwise compromised by magnetism. While we aren't exactly building an MRI, we should have prepared measurements and confidence in lack of hazard - or a warning ready to go. Similarly, we can assume at least some of our geriatric users will have pacemakers, and depending on their age of installation could be compromised by even relatively minor magnetism and frequency interference.
- On a semi related point, we can also assume many of our older users will have hearing aids, which we will need to consider in our Bluetooth protocol among other design factors.

**Cultural and Psychological Factors**

- While the world of wearable biometrics is not new, and is gaining popularity, that has not yet proven to be true of AR glasses or other head-mounted equipment other than hearing aids. We need to be just as subtle and perceived as just as necessary to be successful; passerby should say "that's cool," not cross the street to avoid our users (as I did when those Snapchat glasses were a thing).
- Messaging, branding, and positioning will be a challenge; we are "baking a pie from scratch," so to speak, in an industry that is already perceived as very difficult to understand. Meanwhile we will have a new entry - one that is admittedly better, but that isn't exactly something you can understand without experiencing. We need to cultivate the product identity early and often and make sure messaging, when it comes time, is focused and strengths-based rather than a shotgun blast at 29 different groups. My populations work is development guidance, not a marketing playbook.
- This is a new concept, in form factor and language model and more, entering into an industry driven by "experts" with calcified opinions of what language is and how it should be accessed. A big part of our job between now and launch is that of cementing ourselves as clinical authorities and winning over influencers and evaluators in order to prime the culture for change.
- Recent events, and - well, the nature of humanity and I suppose events for a very long time as well - have shown how toxic misunderstandings can be about technology, particularly biometric technology, and extra particularly when the specter of privacy is raised. We need clear and consistent messaging to defuse any such arguments from the start.
- Cochear Implants are, by consunsus, considered to be disqualifying on the basis of the fact that even when disabled, there is mastoid implant involvment that could compromise signal.
- Finally, to look inwards, I would advocate for team coherence - both within Cognixion and with partners. With multiple products, a rapidly changing sales environment, potential innovation by competitors, and assuredly many different opinions of the definition of "right" (and "done" and many other words), it would be easy to divide into working groups that fail to deliver a cohesive product to the user. Decision hierarchy and overall vision need to be clearly articulated and understood, and - most importantly - maintained consistently by all parties, even when some of them might disagree.

# 3. Final Thoughts

I hope this information was useful to all of you, and thanks to all who took the time to read it. Please feel free to comment, contact me, or otherwise provide feedback in any areas that are lacking or incorrect. Meanwhile, a more brief daily reference can be found [here](https://www.notion.so/0a46504db4b4493a85a1df70cde4628f) - or the longer one [here](https://www.notion.so/8dfb94d7f8f64ecabeb06fd98c5e6a33), for any masochists - and I encourage everyone to check out the clinical videos and personas captured [here](https://www.notion.so/0c3ad1ce57124cc9adbe90d62650e0f0). Citations for all resources are aggregated [here](https://www.notion.so/5720b663ddc544769df1ab7c1f2c30b7) for expediency, if anyone wants the primary information or seeks to verify a statement made.

# Clinical Guidance on the Use of the Cognixion ONE: Human Factors and Clinical Use

This document provides clinical guidance on the use of the Cognixion ONE, a speech generating device funded by Cognixion. The Cognixion ONE is a wearable device that uses augmented reality to provide a communication solution for individuals with various cognitive, motor, and speech disorders. The device aims to assist individuals with communication difficulties in their everyday lives, and it is important to ensure that it is both safe and effective for such use.

The clinical team plays a crucial role in ensuring that the device is safe and effective for use by individuals with cognitive, motor, and speech disorders. They should verify that the target audience for the device is clearly defined and understood. This includes individuals with conditions such as cerebral palsy, amyotrophic lateral sclerosis, and multiple sclerosis.

It is important to understand the exclusions to ensure that trials are conducted safely and effectively. This section outlines the disqualifying symptoms and diagnoses for which a trial is expressly excluded. It is important to note that no trial should occur when conditions from this section are involved without express permission from the Cognixion clinical team and attending medical professionals, even if the user feels otherwise.

## Exclusions

### Exclusion due to Potential Harm

Photosensitive epilepsy, albinism, skin conditions related to specific disorders, allergies, contact dermatitis, and anaphylaxis are disqualifying conditions due to potential harm. These conditions require careful screening and evaluation by a physician before any trials can begin. For instance, photosensitive epilepsy is a neurological disorder that can cause seizures in response to flashing lights, and individuals with this condition may be at risk of experiencing seizures if exposed to the augmented reality elements of the Cognixion ONE.

### Exclusion due to Sensory Inadequacy

Hearing loss, visual impairment, and other sensory inadequacies must be taken into consideration during the design and testing of the device. It is important to determine the extent of visual functioning required for calibration and interaction with augmented reality. For example, individuals with visual impairments may require a different setup for calibration and interaction with the device.

### Exclusion due to Physical Characteristics

Craniofacial abnormalities and morphology, electrode occlusion, or other compromised occipital lobe placement can affect the positioning of electrodes on the head. Brain injuries acquired from outside trauma are excluded from consideration in this work. The Cognixion ONE is designed for an adult population due to the challenges inherent in sizing for children, and pediatric populations have been ruled out, particularly those with average mortality before the end of adolescence.

### Exclusion due to Capacity or Efficacy

Psychosis and other cognitive-perceptual conditions and intellectual disability can affect communication and may require special consideration during the design of the device. For example, individuals with certain cognitive-perceptual conditions may respond differently to visual stimuli, and it may be necessary to create different visual stimuli for different populations.

## Non-Excluding Factors that Warrant Caution

In addition to the exclusions, there are non-excluding factors that warrant caution during the design and testing of the device. These factors should be carefully considered to ensure that the device is safe and effective for use by individuals with cognitive, motor, and speech disorders.

### Content and Experience

Among the potentially included target cohorts are individuals with vastly different needs not just in terms of their condition and level of functioning but also in terms of their environmental context, daily routines, preferred use of language, familiarity with technology, and much more. This has considerable implications for user experience (UX), where a user with aphasia responds to completely different visual stimuli as, say, a 20-year-old girl with Rett syndrome. If we wish to address all of these populations at a high level, we need UX variants that account for that - not to mention training, support, and the out-of-the-box experience.

### Additional Physical Factor Concerns

A portion of the potential thermal hazard is out of our control, in the form of the paired smartphone projecting the visor display. As this population is fragile by nature, we have a high bar in terms of the challenge posed by ventilation - not to mention the expectations users will have around water resistance and inclement weather, which are currently met reasonably well in the tablet-based augmentative and alternative communication industry (~IP67).

### Cultural and Psychological Factors

While the world of wearable biometrics is not new, and is gaining popularity, that has not yet proven to be true of augmented reality glasses or other head-mounted equipment other than hearing aids. We need to be just as subtle and perceived as just as necessary to be successful; passerby should say "that's cool," not cross the street to avoid our users. Messaging, branding, and positioning will be a challenge; we are "baking a pie from scratch," so to speak, in an industry that is already perceived as very difficult to understand. Meanwhile we will have a new entry - one that is admittedly better, but that isn't exactly something you can understand without experiencing. We need to cultivate the product identity early and often and make sure messaging, when it comes time, is focused and strengths-based rather than a shotgun blast at 29 different groups. My populations work is development guidance, not a marketing playbook.

### Cochlear Implants

Cochlear implants are, by consensus, considered to be disqualifying on the basis of the fact that even when disabled, there is mastoid implant involvement that could compromise signal.

## Final Thoughts

In conclusion, this document provides clinical guidance on the use of the Cognixion ONE as a communication solution for individuals with cognitive, motor, and speech disorders. It is important to ensure that the device is safe and effective for use by individuals with these conditions. Exclusions due to potential harm, sensory inadequacy, physical characteristics, and capacity or efficacy should be taken into consideration during the design and testing of the device. Moreover, non-excluding factors that warrant caution include content and experience, additional physical factor concerns, cultural and psychological factors, and cochlear implants. The clinical team should ensure that the device is safe and effective for use by individuals with cognitive, motor, and speech disorders, and verify that the target audience for the device is clearly defined and understood.