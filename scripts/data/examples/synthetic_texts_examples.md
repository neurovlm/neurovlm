# Synthetic texts: re-creation examples

Written by `scripts/data/08_synthetic_texts.py`. The original generation code was not kept, so this run repeats
the process: the synthetic-text prompt (`docs/01_data/07_transform_text_gen.ipynb`, cell 8) with the concept name as
the user message, sent through the Hugging Face Inference router. Each generated text is compared with the released
text of the same name (synth.parquet, text_synth_less.parquet, stage 14 tables) by MiniLM cosine. *Closest* is the
most similar released text of any name, shown when it is not the same-name text.

Cosine to the released text of the same name: 0.84 to 0.92 over 12 texts.
Median cosine of a generated text to all 2,381 released texts: 0.48.

Settings: `{"model": "Qwen/Qwen3.5-27B:cheapest", "provider": "hf-router", "router": "https://router.huggingface.co/v1", "temperature": 0.1, "top_p": 0.9, "max_tokens": 512, "enable_thinking": false, "n_per_seed": 2}`

## network: Limbic

**Sample 1** (model title "Limbic System Overview")

The limbic system comprises a network of structures including the hippocampus, amygdala, and cingulate cortex that are situated deep within the brain. These anatomical regions are frequently associated with processes related to emotional experience, memory formation, and motivational states. The interplay between these areas suggests a role in integrating sensory information with internal emotional responses. While often linked to specific behaviors, the precise functional contributions of each component remain complex and interconnected.

Released text of the same name (synth.parquet, Qwen row), cosine 0.88:

> The limbic network encompasses a collection of interconnected brain structures involved in emotional processing and memory consolidation. Key anatomical components often include the amygdala, hippocampus, and anterior cingulate cortex, which coordinate to regulate affective states. These regions interact with parts of the prefrontal cortex to influence behavioral regulation and social behavior. This system supports complex cognitive functions related to reward and motivation alongside traditional survival mechanisms. Such connectivity illustrates how internal physiological states may shape perception and subsequent actions.

Closest released text: "Limbic Lobe" (synth.parquet, Qwen row), cosine 0.89.

**Sample 2** (model title "Limbic System Overview")

The limbic system comprises a network of structures including the hippocampus, amygdala, and hypothalamus that are situated deep within the brain. These regions are frequently associated with processes involving emotional experience, memory formation, and motivational states. The anatomical connections between these areas suggest a role in integrating sensory information with internal physiological responses. While often linked to affective processing, the precise functional boundaries of this system remain a subject of ongoing investigation.

Released text of the same name (synth.parquet, Qwen row), cosine 0.85:

> The limbic network encompasses a collection of interconnected brain structures involved in emotional processing and memory consolidation. Key anatomical components often include the amygdala, hippocampus, and anterior cingulate cortex, which coordinate to regulate affective states. These regions interact with parts of the prefrontal cortex to influence behavioral regulation and social behavior. This system supports complex cognitive functions related to reward and motivation alongside traditional survival mechanisms. Such connectivity illustrates how internal physiological states may shape perception and subsequent actions.

Closest released text: "Limbic Lobe" (synth.parquet, Qwen row), cosine 0.87.

Cosine between the re-wordings: 0.95.

## network: Auditory Network

**Sample 1** (model title "Auditory Network Overview")

The auditory network encompasses a distributed system of brain regions that process sound information, primarily involving the primary auditory cortex within the superior temporal gyrus. These cortical areas interact with subcortical structures like the medial geniculate nucleus to relay and refine acoustic signals. Broader connections extend to association cortices in the frontal and parietal lobes, which may support the interpretation of complex sounds and language. This interconnected architecture suggests a collaborative role in transforming basic sensory input into meaningful perceptual experiences.

Released text of the same name (synth.parquet, Qwen row), cosine 0.86:

> The auditory network encompasses a distributed system of brain regions involved in the processing of acoustic information. Key structures include the primary auditory cortex located within Heschl’s gyrus and surrounding areas of the superior temporal gyrus. These regions contribute to interpreting sound frequencies and patterns, supporting the distinction between different types of sound. Connections extend to other cortical areas that may integrate auditory input with memory or attention systems. This network represents a fundamental pathway for translating physical sound waves into meaningful neural representations.

Closest released text: "Auditory Association Network" (synth.parquet, Qwen row), cosine 0.87.

**Sample 2** (model title "Auditory Network Overview")

The auditory network encompasses a distributed system of brain regions that process sound information, primarily involving the temporal lobes and the thalamus. Within this system, the primary auditory cortex serves as an initial hub for receiving acoustic signals, while surrounding association areas contribute to interpreting complex features like pitch and rhythm. These anatomical structures interact with broader neural circuits to support functions such as speech perception and environmental awareness. The integration of these regions allows for the continuous monitoring and analysis of auditory input in relation to other sensory experiences.

Released text of the same name (synth.parquet, curated or template row), cosine 0.90:

> Anatomically, the auditory network is built around the superior temporal cortex, encompassing primary auditory cortex and adjacent belt regions. It is broadly associated with analyzing acoustic input across multiple feature dimensions. The network is often described as the foundation of auditory perception. Its coordinated dynamics are thought to support speech, music, and environmental sound processing.

Cosine between the re-wordings: 0.87.

## region: Precentral Gyrus

**Sample 1** (model title "Precentral Gyrus Function")

The precentral gyrus is a prominent ridge located in the frontal lobe, situated immediately anterior to the central sulcus. This anatomical structure contains the primary motor cortex, which is broadly associated with the planning and execution of voluntary movements. Neural signals originating here are thought to coordinate muscle activity across various parts of the body through descending pathways. The organization within this region often reflects a somatotopic map, linking specific cortical areas to distinct muscle groups.

Released text of the same name (synth.parquet, curated or template row), cosine 0.92:

> The precentral gyrus is a frontal lobe region located immediately anterior to the central sulcus. It is commonly associated with voluntary movement and broad motor representation of the body. In fMRI, this region is often described as supporting cortical organization of motor output and action execution.

Closest released text: "Precentral" (synth.parquet, Qwen row), cosine 0.92.

**Sample 2** (model title "Precentral Gyrus Function")

The precentral gyrus is a prominent ridge located in the frontal lobe, situated immediately anterior to the central sulcus. This region contains the primary motor cortex, which is broadly associated with the planning and execution of voluntary movements. Neural signals originating here are thought to coordinate with other cortical areas to facilitate precise motor control across various body parts. The organization within this structure often reflects a somatotopic map, linking specific zones to distinct muscle groups.

Released text of the same name (synth.parquet, curated or template row), cosine 0.92:

> The precentral gyrus is a frontal lobe region located immediately anterior to the central sulcus. It is commonly associated with voluntary movement and broad motor representation of the body. In fMRI, this region is often described as supporting cortical organization of motor output and action execution.

Cosine between the re-wordings: 0.99.

## region: Frontal Medial Cortex

**Sample 1** (model title "Frontal Medial Cortex")

The frontal medial cortex is a region located along the inner surface of the frontal lobes, often considered part of the broader default mode network. This area is frequently associated with internal mental processes such as self-referential thought and the evaluation of social information. Neural activity in this region may reflect the integration of personal memories with current emotional states. It appears to interact with other limbic structures to support complex cognitive functions related to identity and decision-making.

Released text of the same name (synth.parquet, Qwen row), cosine 0.84:

> The frontal medial cortex occupies the inner surface of the frontal lobes, spanning areas near the anterior cingulate and superior frontal gyri. Anatomical pathways link this structure to various limbic regions and nodes within the default mode network. Functional roles associated with this territory include aspects of self-referential processing and social cognition. These neural systems participate in evaluating internal states alongside environmental demands during higher-order thinking.

**Sample 2** (model title "Frontal Medial Cortex")

The frontal medial cortex is a region located along the inner surface of the frontal lobes, often associated with the default mode network. This area appears to participate in processes related to self-referential thought and internal mental simulation. It may also contribute to evaluating social information and integrating personal experiences with broader cognitive contexts. While its precise role remains an active area of inquiry, it is frequently observed during states of rest or introspection.

Released text of the same name (synth.parquet, Qwen row), cosine 0.84:

> The frontal medial cortex occupies the inner surface of the frontal lobes, spanning areas near the anterior cingulate and superior frontal gyri. Anatomical pathways link this structure to various limbic regions and nodes within the default mode network. Functional roles associated with this territory include aspects of self-referential processing and social cognition. These neural systems participate in evaluating internal states alongside environmental demands during higher-order thinking.

Cosine between the re-wordings: 0.90.

## function: Insight

**Sample 1** (model title "Neural Basis of Insight")

Insight involves a sudden reorganization of information that often engages the anterior cingulate cortex and the right anterior superior temporal gyrus. These regions are thought to facilitate the detection of conflicts in current understanding and the integration of distant associations. The process may also rely on broader interactions within the default mode network to support spontaneous idea generation. Such neural dynamics suggest a shift from focused attention to a more diffuse state of processing.

Released text of the same name (synth.parquet, Qwen row), cosine 0.91:

> Insight involves the sudden reorganization of information leading to a new understanding of a situation. Neural patterns associated with this phenomenon often include the anterior cingulate cortex and the hippocampus, along with regions in the right temporal lobe. These areas contribute to forming remote associations and managing conflicting signals during complex cognitive processing. Interactions between the default mode network and frontal control systems likely support the integration required for such realizations.

**Sample 2** (model title "Neural Basis of Insight")

Insight involves the sudden reorganization of information, a process often linked to interactions between the anterior cingulate cortex and the right anterior superior temporal gyrus. These regions appear to facilitate the resolution of cognitive impasses by integrating disparate pieces of information into a coherent solution. The prefrontal cortex may also contribute by monitoring conflict and supporting the flexible shifting of attention required for such realizations. Together, these areas form a network that supports the transition from confusion to clarity without relying on step-by-step analytical reasoning.

Released text of the same name (synth.parquet, Qwen row), cosine 0.86:

> Insight involves the sudden reorganization of information leading to a new understanding of a situation. Neural patterns associated with this phenomenon often include the anterior cingulate cortex and the hippocampus, along with regions in the right temporal lobe. These areas contribute to forming remote associations and managing conflicting signals during complex cognitive processing. Interactions between the default mode network and frontal control systems likely support the integration required for such realizations.

Cosine between the re-wordings: 0.90.

## function: Text Comprehension

**Sample 1** (model title "Neural Basis of Text Comprehension")

Understanding written language involves a distributed network that integrates visual processing with semantic and syntactic analysis. The left inferior frontal gyrus and the posterior middle temporal gyrus are frequently associated with the manipulation of linguistic structure and meaning retrieval. These regions interact with the angular gyrus to support the integration of word forms into coherent conceptual representations. Such connectivity suggests that text comprehension relies on the coordinated engagement of multiple cortical areas rather than a single isolated center.

Released text of the same name (synth.parquet, Qwen row), cosine 0.89:

> Text comprehension relies on a distributed neural network predominantly situated within the left cerebral hemisphere. Visual processing begins in occipital cortices before signals propagate to posterior temporal regions associated with lexical access. Subsequent integration of semantic and syntactic information frequently involves the inferior frontal gyrus and the angular gyrus. These anatomical areas contribute to the synthesis of meaning derived from written linguistic symbols. Functional interactions across these regions support the understanding of complex textual structures.

**Sample 2** (model title "Neural Basis of Text Comprehension")

Understanding written language involves a distributed network that integrates visual processing with semantic and syntactic analysis. The left inferior frontal gyrus and the posterior middle temporal gyrus are frequently associated with the manipulation of linguistic structure and meaning retrieval. These regions interact with the angular gyrus to support the integration of word forms into coherent concepts. Such connectivity suggests that comprehension relies on the coordinated engagement of multiple cortical areas rather than a single isolated center.

Released text of the same name (synth.parquet, Qwen row), cosine 0.87:

> Text comprehension relies on a distributed neural network predominantly situated within the left cerebral hemisphere. Visual processing begins in occipital cortices before signals propagate to posterior temporal regions associated with lexical access. Subsequent integration of semantic and syntactic information frequently involves the inferior frontal gyrus and the angular gyrus. These anatomical areas contribute to the synthesis of meaning derived from written linguistic symbols. Functional interactions across these regions support the understanding of complex textual structures.

Cosine between the re-wordings: 0.99.
