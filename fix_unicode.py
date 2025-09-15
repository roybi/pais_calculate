#!/usr/bin/env python3
"""
Script to fix all Unicode emoji characters in academic_pais.py
"""

# Read the file
with open('academic_pais.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define all Unicode replacements
replacements = {
    '🎓': '[ACADEMIC]',
    '📁': '[FILE]',
    '📊': '[DATA]',
    '📋': '[INFO]',
    '🔄': '[PROCESS]',
    '🔬': '[ANALYSIS]',
    '📅': '[DATE]',
    '🧹': '[CLEAN]',
    '🧠': '[AI]',
    '📡': '[SIGNAL]',
    '🌪️': '[CHAOS]',
    '🔍': '[SEARCH]',
    '📈': '[STATS]',
    '🎯': '[TARGET]',
    '🎭': '[ENSEMBLE]',
    '💥': '[STRONG]',
    '🔨': '[TOOL]',
    '⚠️': '[WARNING]',
    '✅': '[SUCCESS]',
    '❌': '[FAILED]',
    '🚀': '[ENTER]',
    '🏁': '[EXIT]',
    '🧪': '[TEST]',
    '🎉': '[COMPLETE]',
    '🔁': '[LOOP]',
    '📝': '[LOG]',
    '⭐': '[STAR]',
    '💡': '[IDEA]',
    '🎲': '[RANDOM]',
    '📶': '[SIGNAL]',
    '🔮': '[PREDICT]',
    '🧬': '[DNA]',
    '🌊': '[WAVE]',
    '🎨': '[ART]',
    '🎪': '[SHOW]',
    '🔥': '[FIRE]',
    '⚡': '[LIGHTNING]',
    '🌟': '[BRIGHT]',
    '🎬': '[ACTION]',
    '🎵': '[MUSIC]',
    '🏆': '[TROPHY]',
    '🎖️': '[MEDAL]',
    '🏅': '[BADGE]',
    '🔑': '[KEY]',
    '🔓': '[UNLOCK]',
    '🔒': '[LOCK]',
    '🔐': '[SECURE]',
    '🛡️': '[SHIELD]',
    '⚔️': '[SWORD]',
    '🗡️': '[BLADE]',
    '🏹': '[ARROW]',
    '🎯': '[TARGET]',
    '🎱': '[BALL]',
    '🎰': '[SLOT]',
    '🃏': '[JOKER]',
    '🎴': '[CARD]',
    '🀄': '[TILE]',
    '♠️': '[SPADE]',
    '♣️': '[CLUB]',
    '♥️': '[HEART]',
    '♦️': '[DIAMOND]',
}

# Apply all replacements
for emoji, replacement in replacements.items():
    content = content.replace(emoji, replacement)

# Write the fixed content back
with open('academic_pais.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("All Unicode characters have been replaced with plain text equivalents.")