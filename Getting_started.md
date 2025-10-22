# 🚀 HillClimber Quick Start Guide

Selectors pick atoms.
VirtualAtoms reduce groups (COM/COG).
CVs measure things (distance, coordination, etc.).

Three clean layers, one clear mental model. 🧠

---

## 1. Selectors — "Pick What You Want"

Selectors identify groups of atoms in your system.
Every selector returns a list of groups, each a list of atom indices:

[[atoms_of_group_0], [atoms_of_group_1], ...]

**Example:**

```python
ethanol_sel = hc.SMARTSSelector("CCO")  # Matches 2 ethanols
water_sel = hc.SMARTSSelector("O")      # Matches 3 waters
```

---

### Indexing — Two Levels Deep

- **Level 1:** Pick which groups
- **Level 2:** Pick which atoms within those groups

| Syntax | Meaning |
|--------|---------|
| `sel` | All groups |
| `sel[0]` | First group |
| `sel[0:2]` | First two groups |
| `sel[[0, 2]]` | Groups 0 and 2 |
| `sel[:][0]` | First atom of each group |
| `sel[0][0]` | First atom of the first group |

```python
# First oxygen of each water
oxygens = water_sel[:][0]

# First two waters only
first_two = water_sel[0:2]
```

---

### Combining Selectors

Combine any selectors using `+`:

```python
all_mols = water_sel + ethanol_sel
```

→ "Select all waters and ethanols."

---

## 2. VirtualAtoms — "Reduce to COM/COG"

A VirtualAtom creates virtual sites (e.g., centers of mass).

**Key concept:** One virtual site per group in the selector.

```python
water_coms = hc.VirtualAtom(water_sel, "com")  # 3 COMs
ethanol_coms = hc.VirtualAtom(ethanol_sel, "com")  # 2 COMs
```

### Common Patterns

| Code | Meaning |
|------|---------|
| `VirtualAtom(sel, "com")` | COM of each group |
| `VirtualAtom(sel[0], "com")` | COM of the first group only |
| `VirtualAtom(va, "com")` | COM of COMs (nested) |
| `va1 + va2` | Combine multiple COM sets |

**Example:**

```python
# COM for each water
water_coms = hc.VirtualAtom(water_sel, "com")

# COM for ethanol[0]
ethanol_0_com = hc.VirtualAtom(ethanol_sel[0], "com")

# Combine both
all_coms = ethanol_0_com + water_coms
```

💡 **Important:** Indexing happens at the selector, not the VirtualAtom.

---

## 3. CVs — "Measure Things Between Groups or COMs"

CVs use Selectors or VirtualAtoms as inputs.

### DistanceCV

Measures distances between atom groups or virtual sites.

```python
# Distance between ethanol[0] COM and water[0] COM
dist = hc.DistanceCV(
    x1=hc.VirtualAtom(ethanol_sel[0], "com"),
    x2=hc.VirtualAtom(water_sel[0], "com"),
    prefix="d_com"
)
```

**One-to-Many / Many-to-Many:**

```python
# Distance from one ethanol to each water
dist = hc.DistanceCV(
    x1=hc.VirtualAtom(ethanol_sel[0], "com"),
    x2=hc.VirtualAtom(water_sel, "com"),
)
# → Creates 3 CVs: d_0, d_1, d_2
```

---

### CoordinationCV

Measures how "close" groups are — like solvation or clustering.

```python
cn = hc.CoordinationCV(
    x1=hc.VirtualAtom(water_sel, "com"),
    x2=hc.VirtualAtom(water_sel, "com"),
    r_0=5.0,
    prefix="cn_water"
)
# → One CV: coordination among all water COMs
```

---

### Flattening

Controls whether multi-atom groups are flattened or grouped in PLUMED:

| Option | Effect | PLUMED Output |
|--------|--------|---------------|
| `flatten=True` | Atoms merged into one list | `DISTANCE ATOMS=1,2,3,4` |
| `flatten=False` | Each group becomes a GROUP | `G1: GROUP ATOMS=...` |

```python
dist = hc.DistanceCV(
    x1=water_sel[0],
    x2=ethanol_sel[0],
    flatten=False
)
```

---

## 4. Quick Visual Cheat Sheet

```
Selector      → picks atoms or groups
                e.g. water_sel[0:2]

VirtualAtom   → reduces groups (COM/COG)
                e.g. VirtualAtom(water_sel, "com")

CV            → measures between selectors or virtual sites
                e.g. DistanceCV(x1, x2)
```

---

## 5. Common Real-World Examples

| Goal | Code | Output |
|------|------|--------|
| Distance between two specific atoms | `hc.DistanceCV(x1=sel1[0][0], x2=sel2[0][0])` | `DISTANCE ATOMS=1,2` |
| Distance between molecule COMs | `hc.DistanceCV(x1=hc.VirtualAtom(sel1[0],"com"), x2=hc.VirtualAtom(sel2[0],"com"))` | `COM ATOMS=...` |
| Distance from one COM to all COMs | `hc.DistanceCV(x1=va1[0], x2=va2)` | Multiple CVs |
| Coordination between all COMs | `hc.CoordinationCV(x1=va, x2=va, r_0=5.0)` | One CN CV |
| COM of COMs (e.g., cluster center) | `hc.VirtualAtom(hc.VirtualAtom(sel, "com"), "com")` | Single virtual site |

---

## 6. Mental Model

| You Think | You Write | PLUMED Thinks |
|-----------|-----------|---------------|
| "Select these atoms" | `selector` | atom indices |
| "Group them by molecule" | `selector[0:2]` | grouped atoms |
| "Compute their COM" | `VirtualAtom(selector, "com")` | `COM ATOMS=...` |
| "Measure a distance" | `DistanceCV(x1, x2)` | `DISTANCE ATOMS=...` |
| "Compute coordination" | `CoordinationCV(x1, x2)` | `COORDINATION GROUPA=...,GROUPB=...` |

---

## 7. Golden Rules

1. **Selectors select.**
   - Always use selectors for picking atoms and groups.
2. **VirtualAtoms reduce.**
   - Use them only for COM/COG — never for indexing.
3. **CVs measure.**
   - They combine selectors or virtual sites to produce observables.
4. **Index at the selector level.**
   - e.g., `VirtualAtom(sel[0], "com")`, not `VirtualAtom(sel, "com")[0]`.
5. **Combine with `+`.**
   - `sel1 + sel2` or `va1 + va2` to merge sets.
6. **Flatten mindfully.**
   - `flatten=True` → single atom list
   - `flatten=False` → PLUMED GROUPs

---

## 8. Typical PLUMED Translations

| Python Expression | PLUMED Output |
|-------------------|---------------|
| `hc.VirtualAtom(water_sel, "com")` | `water_0_com: COM ATOMS=...` |
| `hc.DistanceCV(x1=va1, x2=va2)` | `DISTANCE ATOMS=va1_0,va2_0` |
| `hc.CoordinationCV(x1=va, x2=va, r_0=5.0)` | `COORDINATION GROUPA=va_0,... GROUPB=va_0,... R_0=5.0` |

---

## 9. Summary of Best Practices

| Task | Recommended Approach |
|------|---------------------|
| Select atoms or molecules | SMARTSSelector, use indexing |
| Combine multiple groups | `sel1 + sel2` |
| Create COM/COG | `VirtualAtom(sel, "com" / "cog")` |
| Compute distances | `DistanceCV(x1, x2)` |
| Compute coordination numbers | `CoordinationCV(x1, x2)` |
| Preserve group structure | `flatten=False` |
| Build hierarchical centers | `VirtualAtom(VirtualAtom(sel, "com"), "com")` |

---

## TL;DR

Think modular:
- **Selectors** → choose what
- **VirtualAtoms** → define where
- **CVs** → measure how

No hidden behavior. No implicit COMs. Just clean, explicit, composable logic — fully PLUMED-compatible.