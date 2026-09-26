# Person-to-PPE Spatial Association Algorithm

## 1. Executive Overview

In computer vision-based safety monitoring, object detectors typically output disconnected bounding boxes for generic classes (e.g., `Person`, `Hardhat`, `Safety Vest`, `NO-Hardhat`, `NO-Safety Vest`). 

To assess compliance accurately, the system must solve the **Person-to-PPE Association Problem**:
> *Given a set of detected person bounding boxes and a set of detected PPE bounding boxes in a single frame, determine which PPE items belong to each individual.*

The [`detection/ppe_association.py`](file:///d:/ppe-compliance/detection/ppe_association.py) module implements a **deterministic geometric association engine** utilizing anatomical body proportions, containment metrics, and multi-factor scoring to achieve real-time, low-overhead matching without requiring heavy secondary neural networks.

---

## 2. Association Pipeline

```
  +-----------------------+        +---------------------+
  |   Person Detections   |        |   PPE Detections    |
  +-----------------------+        +---------------------+
              \                              /
               \                            /
                v                          v
       +---------------------------------------------+
       |       Geometric Filtering & Regional        |
       |             Anatomical Validation           |
       |  (Head: 0-38% height | Torso: 12-72% height)|
       +---------------------------------------------+
                             |
                             v
       +---------------------------------------------+
       |        Multi-Factor Candidate Scoring       |
       |   Score = w_cont * Cont + w_prox * Prox     |
       |                 + w_conf * Conf             |
       +---------------------------------------------+
                             |
                             v
       +---------------------------------------------+
       |   Disambiguation & Exclusive Assignment    |
       |        (Best-Scoring Person Wins)           |
       +---------------------------------------------+
                             |
                             v
       +---------------------------------------------+
       |       Structured Output & Conversion        |
       |      (AssociatedPerson -> PersonPPEInfo)    |
       +---------------------------------------------+
```

---

## 3. Mathematical Formulation

### 3.1. Relative Coordinate Normalization

Let a detected person bounding box be $B_p = (x_{p1}, y_{p1}, x_{p2}, y_{p2})$ with width $W_p = x_{p2} - x_{p1}$ and height $H_p = y_{p2} - y_{p1}$.
Let a candidate PPE bounding box be $B_{ppe} = (x_1, y_1, x_2, y_2)$ with center:

$$cx = \frac{x_1 + x_2}{2}, \quad cy = \frac{y_1 + y_2}{2}$$

The relative vertical ($rel\_y$) and horizontal ($rel\_x$) positions within the person's frame are:

$$rel\_y = \frac{cy - y_{p1}}{H_p}, \quad rel\_x = \frac{cx - x_{p1}}{W_p}$$

### 3.2. Anatomical Prior Regions

Human body proportions dictate where specific safety gear is physically worn:

1. **Head Region** (`Hardhat`, `NO-Hardhat`, `Mask`, `NO-Mask`):
   - Valid vertical window: $rel\_y \in [head\_min\_y, head\_max\_y]$ (default: `[-0.20, 0.38]`).
   - The negative bound ($-0.20$) accounts for hardhats protruding above the top of the detected person's head box.
   - Ideal anatomical anchor: $ideal\_y = 0.08$.

2. **Torso Region** (`Safety Vest`, `NO-Safety Vest`):
   - Valid vertical window: $rel\_y \in [torso\_min\_y, torso\_max\_y]$ (default: `[0.12, 0.72]`).
   - Ideal anatomical anchor: $ideal\_y = 0.38$.

3. **Horizontal Boundaries**:
   - The PPE centroid must satisfy $rel\_x \in [-margin_x, 1.0 + margin_x]$ (default margin: $20\%$).

### 3.3. Multi-Factor Association Scoring

For every valid (person, PPE) candidate pair, a composite score $S \in [0.0, 1.0]$ is computed:

$$S = w_{containment} \cdot C + w_{proximity} \cdot P + w_{confidence} \cdot Conf$$

Where:
- **Containment Ratio ($C$)**: The fraction of the PPE bounding box contained inside the person's box:
  $$C = \frac{\text{Area}(B_p \cap B_{ppe})}{\text{Area}(B_{ppe})}$$
- **Proximity Score ($P$)**: Evaluates alignment with the central vertical axis and the anatomical ideal height:
  $$P = 0.5 \cdot \left(1.0 - \frac{|rel\_x - 0.5| \cdot 2}{1 + 2 \cdot margin_x}\right) + 0.5 \cdot \left(1.0 - \frac{|rel\_y - ideal\_y|}{\text{half\_span}_y}\right)$$
- **Confidence ($Conf$)**: Detector confidence score for the PPE item.

### 3.4. Exclusive 1-to-1 Disambiguation

If a PPE detection satisfies the spatial criteria for multiple persons (e.g., workers standing close together), the PPE item is assigned **exclusively** to the person with the **highest association score**:

$$Person^* = \arg\max_{p \in \text{Candidates}} S(p, PPE)$$

Any PPE item failing to meet the `min_association_score` threshold ($0.20$) remains unassigned (e.g., a spare hardhat placed on equipment).

---

## 4. Configurable Parameters

All thresholds are exposed through the [`AssociationConfig`](file:///d:/ppe-compliance/detection/ppe_association.py) dataclass:

| Parameter | Default Value | Description |
| :--- | :---: | :--- |
| `head_min_rel_y` | `-0.20` | Lower vertical limit for headgear relative to person height |
| `head_max_rel_y` | `0.38` | Upper vertical limit for headgear |
| `head_ideal_rel_y` | `0.08` | Center line for headgear alignment |
| `torso_min_rel_y` | `0.12` | Upper vertical limit for torso items |
| `torso_max_rel_y` | `0.72` | Lower vertical limit for torso items |
| `torso_ideal_rel_y` | `0.38` | Center line for vest alignment |
| `horizontal_margin_ratio` | `0.20` | Allowed horizontal leeway outside person box width |
| `min_association_score` | `0.20` | Minimum score threshold to accept a match |
| `weight_containment` | `0.45` | Weight for bbox containment ratio |
| `weight_proximity` | `0.40` | Weight for anatomical proximity |
| `weight_confidence` | `0.15` | Weight for detector confidence |

---

## 5. Assumptions

1. **Upright Posture**:
   The primary assumption is that workers are standing, walking, or operating machinery in an upright or slightly leaning posture.
2. **Camera Perspective**:
   The camera angle is roughly eye-level or slight high-angle CCTV ($< 45^\circ$ elevation angle). Extreme bird's-eye views compress the vertical span and require tuned thresholds.
3. **Detector Consistency**:
   The underlying YOLOv8 detector provides bounding boxes that encompass a majority of the person's body.

---

## 6. Limitations & Edge Cases

| Scenario | Challenge | Mitigating Strategy |
| :--- | :--- | :--- |
| **Crowded / Overlapping Workers** | Two people standing in front of each other share pixel space. | The containment and proximity scoring selects the person whose center aligns closest with the item. |
| **Bending / Crouching Workers** | When a worker bends over, the head drops to mid-torso height. | Future integration with 2D Pose Estimation (e.g., YOLOv8-pose keypoints) will replace box priors with skeleton joints. |
| **Partial Occlusion (Frame Edge)** | A worker's lower body is cut off by the edge of the frame. | Relative coordinates use visible person box; head and upper torso remain valid. |
| **Conflicting Labels on One Person** | Model detects both `Hardhat` and `NO-Hardhat` for the same head. | `AssociatedPerson` resolves ties by comparing detection confidence scores. |

---

## 7. Integration with Compliance Engine

The output of `PPEAssociator.associate()` produces `AssociatedPerson` instances that provide an automatic factory method:

```python
from detection.ppe_association import PPEAssociator
from compliance.rules import ComplianceEngine

# 1. Run geometric association
associator = PPEAssociator()
result = associator.associate(person_detections, ppe_detections)

# 2. Evaluate compliance for each person
engine = ComplianceEngine()
for person in result.persons:
    compliance_input = person.to_person_ppe_info()
    compliance_report = engine.evaluate(compliance_input)
    print(f"{person.person_id}: {compliance_report.summary}")
```
