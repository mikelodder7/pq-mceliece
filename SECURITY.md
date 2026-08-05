# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.3.x   | yes       |
| 0.2.x   | no        |
| 0.1.x   | no        |

## Reporting a Security Bug

If you think you have discovered a security issue in any of the code, I'd love to hear from
you. I take all security bugs seriously; once investigation confirms a bug, I will patch it
within a reasonable amount of time, release a public security bulletin discussing the impact,
and credit the discoverer.

The best way to report a security bug is to email a description of the flaw and any related
information (e.g. reproduction steps, affected version) to the author at
<redmike7@gmail.com>, or to use
[GitHub private vulnerability reporting](https://github.com/mikelodder7/pq-mceliece/security/advisories/new)
on this repository. Please do not open a public issue for a suspected vulnerability before
contacting me privately.

## Scope

This crate aims for constant-time behavior at the source level; `CONFORMANCE.md` documents
exactly what is and is not claimed, including the declassified conditions the reference
implementation also treats as public. Reports that break a documented claim are in scope.
Physical side channels (power, electromagnetic, fault injection) are documented as out of
scope for this implementation.

This crate has not been independently audited.
