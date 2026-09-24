# Storyboard

**Format:** 1920x1080  
**Audio:** Voiceover if generated; otherwise visual-first promo with the same script as on-screen timing reference.  
**VO direction:** Calm technical presenter, confident and human, short pauses between statements.  
**Style basis:** `DESIGN.md` and captured source UI.

## Asset Audit

| Asset | Type | Assign to Beat | Role |
| --- | --- | --- | --- |
| DOM/CSS protocol stage | Source motif | Beats 1, 2, 5 | Main product signal |
| Client entity card | Source motif | Beats 1, 2, 3, 5 | Browser endpoint |
| Server entity card | Source motif | Beats 1, 2, 3, 5 | Site endpoint |
| Packet rail and packet cards | Source motif | Beats 1, 2, 4 | Handshake motion |
| Key rows and certificate material | Source motif | Beat 3 | Trust and shared-secret proof |
| Code panel / ciphertext | Source motif | Beat 4 | Plain-to-encrypted transformation |
| External downloaded assets | None | Skip | Capture found no external assets |

## Beat 1 - Make TLS Visible (0.00-3.60s)

**VO cue:** "TLS is invisible, until you can watch it move."  
**Concept:** The viewer lands inside the demo's lab canvas. The product promise appears first: an invisible network protocol can be watched, stepped through, and understood.  
**Visual:** Pale grid background. Big headline at left. A compact client-to-server diagram at right. A packet card launches across the rail while tiny ClientHello metadata lines wake up underneath it.  
**Techniques:** Kinetic typography, SVG path drawing, packet motion.  
**Transition:** Blur-through into Beat 2.

## Beat 2 - Live Handshake Map (3.60-7.60s)

**VO cue:** "This interactive demo turns every handshake into a live map: client, server, packet, key."  
**Concept:** The UI expands into a product tour. The important claim is not "TLS explained" but "TLS mapped."  
**Visual:** Client and Server cards lock into opposite sides. A vertical step rail fills through the nine TLS moments. Packet chips cascade down the center. The active step row punches forward.  
**Techniques:** Step counter fill, staggered packet cards, color-coded ownership.
**Transition:** Push/blur upward into Beat 3.

## Beat 3 - Proof, Not Hand Waving (7.60-11.80s)

**VO cue:** "Follow ClientHello, certificate proof, E C D H E secrets, and Finished checks."  
**Concept:** Trust becomes visible. The beat shows that the demo covers the security mechanism, not only message names.  
**Visual:** Certificate card, signed transcript line, two key-share rows, and a central shared secret. Verification checkmarks draw in sequence from certificate to Finished.  
**Techniques:** SVG path drawing, data-row highlighting, checkmark reveal.
**Transition:** Grid-like data wipe into Beat 4.

## Beat 4 - Plain To Protected (11.80-16.00s)

**VO cue:** "Then see plain H T T P become authenticated encrypted data."  
**Concept:** The payoff: the reason the handshake exists. Plain request text compresses into TLS application data and emerges as authenticated ciphertext blocks.  
**Visual:** Left side starts as `GET /` with headers. The request passes through an AEAD lock and becomes grouped ciphertext blocks. A green authentication tag seals the bottom.  
**Techniques:** Typing effect, block cascade, lock pulse.
**Transition:** Gentle blur crossfade into final CTA.

## Beat 5 - CTA (16.00-20.00s)

**VO cue:** "Teach TLS without hand waving."  
**Concept:** Close as a product promo for the interactive demo. The final frame is useful and inspectable, not abstract.  
**Visual:** Full protocol stage returns, simplified and balanced. Headline reads "Teach TLS without hand waving." CTA button says "Open the interactive demo." Client/server colors stay visible in the progress rail.  
**Techniques:** Final title build, progress rail sweep, subtle panel breathing.
**Transition:** Final fade to canvas color in the last half second.

## Production Architecture

```text
tls-product-promo/
├── index.html
├── DESIGN.md
├── SCRIPT.md
├── STORYBOARD.md
├── narration.txt
├── transcript.json
└── capture/
    ├── screenshots/
    └── extracted/
```
