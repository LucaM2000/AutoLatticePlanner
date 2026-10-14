To install:
pip install auto-lattice-planner

To run:
run-auto-lattice-planner-gui

# AutoLatticePlanner

**AutoLatticePlanner** is an open-source software tool for automated lattice sphere placement in **Lattice Radiotherapy (LRT)** treatment planning. The software optimises the position of a predefined lattice geometry within patient-specific anatomy, generating RTSTRUCT files that can be imported into compatible treatment planning systems (TPSs) for dose optimisation and treatment planning.

The program was developed to reduce the time, variability, and manual effort associated with conventional lattice sphere placement while preserving established lattice geometries used in clinical practice.

## Features
- Automated lattice sphere placement within a target volume
- Grid-search optimisation of lattice rotation and translation
- Deterministic and reproducible sphere placement
- Automatic exclusion of spheres extending outside the target volume
- Optional optimisation towards biologically relevant sub-volumes (e.g. hypoxic regions)
- User-defined organ-at-risk (OAR) avoidance margins
- Standard DICOM RTSTRUCT export
- TPS-independent workflow
- Open-source and freely available

## Method

AutoLatticePlanner uses a structured search of lattice orientations combined with greedy translational refinement to identify lattice positions which maximise the number of valid spheres contained within the target volume.

Unlike methods that optimise sphere diameter, spacing, or individual sphere locations, AutoLatticePlanner preserves a predefined lattice geometry and instead optimises its placement relative to patient anatomy. This approach maintains consistency with established clinical lattice designs while improving geometric coverage of irregular target volumes.

## Workflow

- Import patient DICOM and RTSTRUCT data.
- Select target volume(s).
- Define optional OAR exclusion margins.
- Select lattice geometry size.
- Run automated optimisation.
- Export the generated lattice as a DICOM RTSTRUCT file.
- Import the RTSTRUCT into the TPS for dose optimisation and plan generation.


## Advantages

Compared with manual lattice sphere placement, AutoLatticePlanner:

- Reduces sphere-placement time from approximately 20–30 minutes to under 2 minutes.
- Eliminates manual removal of partial spheres.
- Removes inter-operator variability.
- Produces deterministic and reproducible results.
- Increases the number of valid spheres placed within target volumes.
- Operates independently of proprietary treatment-planning software.


## Validation

The software was evaluated on a cohort of 20 sarcoma patients and compared against a manual lattice planning workflow. Automated placement:

- Increased median sphere count by approximately 30%.
- Increased mean target dose, V12, and EUD.
- Reduced planning time to under 2 minutes (varies depending on target volume size).
- Produced more consistent inter-sphere dose distributions.

## Requirements

Python 3.12

Additional dependencies may be required depending on the version being used.

## Citation

If you use AutoLatticePlanner in research, please cite:

Mulcahy L, Sunyach MP, Baudier T, et al. Automatic Vertex Placement for Lattice SFRT Treatment Plans. (under review).


## Disclaimer

AutoLatticePlanner is intended for research and development purposes. Clinical use should be performed in accordance with local regulations and institutional quality assurance procedures.
