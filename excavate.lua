require 'torch'
require 'nn'

local cjson = require 'cjson'
local cjson2 = cjson.new()
local pretty = require "resty.prettycjson"

utf8 = require 'lua-utf8'

require 'LanguageModel'

local GARBAGE_INTERVAL = 100

-- version of sample which passes in a coroutine to mess with 
-- the probability weights


-- try doing the punctuation comparison by index rather than token
-- I think it's a wide character issue


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'Projects/Musketeers/Musketeers2/Musketeers2_cp_176000.t7')
cmd:option('-vocab', 'aom_vocab.json')
cmd:option('-notpunct', '’')
--cmd:option('-notpunct', '')
cmd:option('-suppress', '')
cmd:option('-excavate', 100)
cmd:option('-alliterate', '')
cmd:option('-length', 1000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', .3)
cmd:option('-name', 'excavate')
cmd:option('-outdir', '/Users/mike/Desktop/NaNoGenMo2019/Samples/')

local END_OFFSET = 5

local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local tokens = {}
local punctuation = {}



model:evaluate()

local punctpat = nil
if opt.notpunct ~= '' then
  punctpat = '[' .. opt.notpunct .. ']'
end

for idx, token in pairs(model.idx_to_token) do
  tokens[idx] = 1
  if token:match('%W') then
    if not punctpat or not (utf8.match(token, punctpat)) then 
      punctuation[idx] = 1
    end
  end
end




local wmap = {}
local words = {}


local f = io.open(opt.vocab, "r")
local vjson = f:read("*all")
local suppressPat = nil
if #opt.suppress > 0 then
  suppressPat = '[' .. opt.suppress .. ']'
end

local vocabj = cjson2.decode(vjson)


for _, iw in pairs(vocabj['words']) do
  local w = iw[1]
  local i = iw[0]
  if w ~= "" then
    if suppressPat then
      if not w:find(suppressPat) then
        -- wmap[w] = 1
        words[#words + 1] = iw
      end
    else
      -- wmap[w] = 1
      words[#words + 1] = iw
    end
  end
end

vocabj = nil

collectgarbage()
print("Loaded JSON", collectgarbage('count') * 1024)

function utf8first(s)
  local o2 = utf8.offset(s, 1)
  if o2 == nil then
    return s
  else
    return s:sub(1, o2 - 1)
  end
end







function get_matches(ws)
  local matches = {}
  for p, _ in pairs(punctuation) do
    matches[p] = 1
  end
  if ws then
    for _, w in pairs(ws) do
      if w then
        local idx = model.token_to_idx[utf8first(w)]
        if idx ~= nil then
          matches[idx] = 1
        else
          -- print("unknown first char in '" .. w .. "'")
          -- print("'" .. utf8first(w) .. "'")
        end
      end
    end
  end
  return matches
end




function matches_to_weights(matches)
  local weights = {}
  for idx, _ in pairs(tokens) do
    if matches[idx] ~= nil then
      weights[idx] = 1
    else
      weights[idx] = 0
    end
  end
  return weights
end


-- input to this is a list of [ index, words ]
-- output is just words

function init_vocab_orig(ws)
  local v = {}
  for i, iw in pairs(ws) do
    v[i] = iw[2]
  end
  return v
end


-- lookahead = { unpack(words, index, index + MAX_AHEAD - 1) }

function init_vocab(ws, start, nwords)
  local v = {}
  for i = start, start + nwords - 1 do 
    v[i - start + 1] = ws[i][2]
  end
  return v
end



function prune_vocab(ov, next_char)
  local v = { "\n", " " }
  for _, w in pairs(ov) do
    local f = utf8first(w)
    if next_char == f then
      local n = w:sub(#f + 1, #w)
      if n then
        v[#v+1] = n
      end
    end
  end
  return v
end


function make_vocab(vocab_gen)
  return coroutine.create(function(prev_char)
    local ok, vocab = coroutine.resume(vocab_gen, nil)
    local current_word = ''
    while ok
      do
        local weights = {}
        local matches = get_matches(vocab)
        local weights = matches_to_weights(matches)
        p = coroutine.yield(weights)
        local next_idx = p[{1,1}]
        local next_char = model.idx_to_token[next_idx]
        if punctuation[next_idx] then
          -- print('"' .. current_word .. '" -> "' .. next_char .. '"')
          vocab = nil
          ok, vocab = coroutine.resume(vocab_gen, current_word)
          current_word = ''
        else
          current_word = current_word .. next_char
          vocab = prune_vocab(vocab, next_char)
          if #vocab < 1 then
            vocab = nil
            ok, vocab = coroutine.resume(vocab_gen, nil)
          end
        end
        matches = nil
        weights = nil
      end
      error("Vocabulary exhausted")
  end)
end


local basic_vocab = coroutine.create(function(used_word)
  while true do
    coroutine.yield(init_vocab(words, 1, #words))
  end
end)


local MAX_AHEAD = 500

local word_indices = {}

-- now words is an array of [i, w] where i is the index from the
-- vocab list.

-- init_vocab strips the indices out and gives the RNN code just words
-- and the matching in this bit tries to reinstate them

local excavate_vocab = coroutine.create(function(used_word)
  local used_word = nil
  local index = 1
  while index <= #words do
    if used_word ~= nil then
      while index <= #words and not utf8.match(words[index][2], '^' .. used_word) do
        index = index + 1
      end
      if #used_word > 0 then
        word_indices[ #word_indices + 1 ] = { words[index][1], used_word }
      end
    end
    if index <= #words then
      -- I suspect this is where it's leaking
      -- local lookahead = { unpack(words, index, index + MAX_AHEAD - 1) }
      local newv = init_vocab(words, index, MAX_AHEAD)
      used_word = coroutine.yield(newv)
      if index % GARBAGE_INTERVAL == 0 then
        print("Memory: ", collectgarbage("count") * 1024)
        collectgarbage()
      end
    end
  end
  print("Vocabulary finished") 
end)

function make_alliterate(char)
  return coroutine.create(function(prev_char)
    local first_t = {}
    first_t[model.token_to_idx[char]] = 1
    first_t[model.token_to_idx[char:upper()]] = 1
    first_t[model.token_to_idx[' ']] = 1
    local weight_t = matches_to_weights(first_t)
    local weights = weight_t
    while true
      do
        p = coroutine.yield(weights)
        local next_char = model.idx_to_token[p[{1,1}]]
        if next_char:match("%W") then
          weights = weight_t
        else
          weights = {}
        end
      end
  end)
end


print("Memory: ", collectgarbage("count") * 1024)

collectgarbage()

local sample = nil

if opt.alliterate ~= '' then
  local mod = make_alliterate(opt.alliterate:sub(1,1))
  sample = model:sample_hacked(opt, mod)
else
  if opt.excavate ~= 0 then
    MAX_AHEAD = opt.excavate
    local tuner = make_vocab(excavate_vocab)
    sample = model:sample_hacked(opt, tuner)
  else
    local tuner = make_vocab(basic_vocab)
  sample = model:sample_hacked(opt, tuner)
  end
end

print("Memory: ", collectgarbage("count") * 1024)


opt['wordcount'] = #word_indices

local jsonfilename = opt.outdir .. '/' .. opt.name .. '.json'
local textfilename = opt.outdir .. '/' .. opt.name .. '.txt'

local jsonfile = io.open(jsonfilename, "w")
jsonfile:write(pretty({ words = word_indices, settings = opt }))
jsonfile:close()

print("Wrote word indices to " .. jsonfilename)

local textfile = io.open(textfilename, "w")
textfile:write(sample)
textfile:close()

print("Wrote text output to " .. textfilename)


