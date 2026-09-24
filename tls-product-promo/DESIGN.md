# Design System

## Overview

The TLS demo uses a light technical workspace: pale grid background, white bordered panels, compact controls, and protocol entities arranged like a lab diagram. The visual identity is instructional and precise, with clear color ownership for client, server, network, and verification state. It avoids decorative imagery; the recognizable product signal is the moving handshake diagram and the side-by-side client/server state model.

## Colors

- **Canvas**: `#F7F8F6` - pale grid background.
- **Panel**: `#FFFFFF` - cards, protocol boxes, and UI surfaces.
- **Ink**: `#16201F` - primary headings, active controls, dark code panels.
- **Muted Text**: `#5F6D69` - secondary explanation text.
- **Border**: `#D9DFDB` - panel outlines and dividers.
- **Client**: `#0D9488` - browser/client identity and outbound material.
- **Server**: `#D94645` - server/site identity and certificate material.
- **Network**: `#577590` - packet rail, transport messages, timeline markers.
- **OK**: `#2F8F4E` - verification and successful shared-secret states.
- **Warn**: `#B26A00` - caution and contrast callouts.
- **Code Surface**: `#101816` - toy handshake code panel.

## Typography

- **Primary Sans**: `ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`. Used for display, labels, and body text. Original hero is 50px/820; promo scales this to video-sized 76-118px.
- **Monospace**: `"SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace`. Used for packet metadata, secrets, code, and ciphertext. Use tabular numbers for all data labels.
- **Hierarchy**: Large statements use heavy sans. Protocol and data fragments use monospace. Body copy stays 24px+ for rendered readability.

## Elevation

Depth comes from thin borders and soft shadows, not heavy 3D. Source panels use `1px` borders in `#D9DFDB`, `8px` radius, and a broad shadow like `0 18px 50px rgba(31, 45, 42, 0.12)`. Promo scenes can add subtle layered panel offsets, but should keep the crisp lab-diagram feel.

## Components

- **Protocol Stage**: Large white canvas with left client lane, right server lane, and a central packet rail.
- **Entity Cards**: Client and Server boxes with colored borders and compact state chips.
- **Packet Cards**: White rounded message boxes with bold title and monospace metadata.
- **Key Rows**: Monospace bordered rows showing randoms, key shares, and traffic secrets.
- **Step List**: Numbered handshake steps with one active dark row.
- **Code Panel**: Dark monospace block explaining the toy shared-secret calculation.
- **Progress Rail**: Thin track using client-to-server color progression.

## Do's and Don'ts

### Do's

- Use exact source colors for client, server, network, and ink.
- Keep the grid, rails, packet cards, and state chips as the recognizable visual language.
- Use motion to clarify protocol direction: packets travel, secrets derive, checks lock.
- Keep panels sharp with `8px` radius and restrained shadows.
- Use monospace for all protocol artifacts and ciphertext.

### Don'ts

- Do not introduce unrelated characters, mascots, 3D objects, or brand-new product metaphors.
- Do not use dark full-screen gradient themes; the source is a light lab canvas.
- Do not add extra protocol entities beyond client, server, network, certificate, and keys.
- Do not turn the demo into a generic cybersecurity ad.
- Do not use purple/blue SaaS gradients or stock-like imagery.
