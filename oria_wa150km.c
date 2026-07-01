/** @file
    Oria WA150KM temperature/humidity sensor decoder.

    Copyright (C) 2025 Jan Niklaas Wechselberg
    Humidity decode added 2026.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
*/
#include "decoder.h"

/**
Oria WA150KM temperature/humidity sensor decoder.

Manchester coding (G.E. Thomas), bit-reflected.

Data layout after decoding:
    0  1  2  3  4  5  6  7  8  9 10 11 12 13
    FF FF FF MM ?? CC DD TT II SH HH XX CK BB

- FF = Preamble: 3 bytes of 0xff
- MM = Message type (toggles fa20/fa28 each send; fa21 on TX button)
- CC = Channel (upper nibble + 1)
- DD = Device ID
- TT = Temperature decimal (upper nibble)
- II = Temperature integer (BCD)
- SH = byte 9: low nibble = sign/flags (bit 3 = negative temperature),
               HIGH nibble = humidity ONES digit
- HH = byte 10: low nibble = humidity TENS digit (high nibble fixed 0xA)
- XX = byte 11: unknown (varies; possibly battery / status)
- CK = byte 12: checksum (varies pseudo-randomly)
- BB = Fixed value 0x65

Humidity = (byte10 & 0x0F) * 10 + (byte9 >> 4)   [verified against sensor LCD 54..89% RH]
*/

#define ORIA_WA150KM_BITLEN 227

static int oria_wa150km_decode(r_device *decoder, bitbuffer_t *bitbuffer)
{
    // Find a valid row (skipping short preamble rows)
    int r;
    for (r = 0; r < bitbuffer->num_rows; r++) {
        if (bitbuffer->bits_per_row[r] == ORIA_WA150KM_BITLEN) {
            break;
        }
    }
    if (r >= bitbuffer->num_rows) {
        decoder_logf(decoder, 2, __func__, "No valid row found with %d bits", ORIA_WA150KM_BITLEN);
        return DECODE_ABORT_LENGTH;
    }

    // Check warmup bytes before decoding
    uint8_t *b = bitbuffer->bb[r];
    if (b[0] != 0xAA || b[1] != 0xAA || b[2] != 0xAA) {
        decoder_log(decoder, 2, __func__, "Warmup bytes are not 0xaaaaaa");
        return DECODE_ABORT_EARLY;
    }

    // Check last byte (raw data before Manchester decoding)
    if (b[bitbuffer->bits_per_row[r] / 8 - 1] != 0x69) {
        decoder_log(decoder, 2, __func__, "Last byte is not 0x69");
        return DECODE_ABORT_EARLY;
    }

    // Invert the buffer for G.E. Thomas decoding
    bitbuffer_invert(bitbuffer);

    // Manchester decode the row
    bitbuffer_t manchester_buffer = {0};
    bitbuffer_manchester_decode(bitbuffer, r, 0, &manchester_buffer, ORIA_WA150KM_BITLEN);

    // Reflect bits in each byte
    reflect_bytes(manchester_buffer.bb[0], (manchester_buffer.bits_per_row[0] + 7) / 8);

    b = manchester_buffer.bb[0];

    // Channel (upper nibble + 1)
    uint8_t channel = ((b[5] >> 4) & 0x0F) + 1;

    // Device ID
    uint8_t device_id = b[6];

    // Temperature: BCD integer in b[8], decimal in high nibble of b[7]
    float temperature = (((b[8] >> 4) & 0x0F) * 10 + (b[8] & 0x0F)) + ((b[7] >> 4) & 0x0F) * 0.1f;
    if (b[9] & 0x08) { // sign bit (bit 3 of b[9] low nibble)
        temperature = -temperature;
    }

    // Humidity: tens digit = low nibble of b[10], ones digit = high nibble of b[9]
    int humidity = (b[10] & 0x0F) * 10 + ((b[9] >> 4) & 0x0F);

    /* clang-format off */
    data_t *data = data_make(
            "model",        "",     DATA_STRING, "Oria-WA150KM",
            "id",           "",     DATA_INT,    device_id,
            "channel",      "",     DATA_INT,    channel,
            "temperature_C","",     DATA_FORMAT, "%.1f C", DATA_DOUBLE, temperature,
            "humidity",     "",     DATA_FORMAT, "%d %%",  DATA_INT,    humidity,
            NULL);
    /* clang-format on */

    decoder_output_data(decoder, data);
    return 1;
}

static char const *const output_fields[] = {
        "model",
        "id",
        "channel",
        "temperature_C",
        "humidity",
        NULL,
};

r_device const oria_wa150km = {
        .name        = "Oria WA150KM freezer and fridge thermometer",
        .modulation  = OOK_PULSE_PCM,
        .short_width = 490,
        .long_width  = 490,
        .gap_limit   = 1500,
        .reset_limit = 4000,
        .decode_fn   = &oria_wa150km_decode,
        .priority    = 10,
        .fields      = output_fields,
};
