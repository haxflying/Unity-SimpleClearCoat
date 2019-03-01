Shader "Custom/ClearCoat" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_ClearCoat_Smoothness ("ClearCoat Smoothness", Range(0, 1)) = 0.6
		_ClearCoat_Metallic("ClearCoat Metallic",Range(0, 1)) = 0.5
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
		//_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", 2D) = "black" {}
		_NormalMap("Normal Map", 2D) = "black" {}
	}
	SubShader {
		Tags { "RenderType"="Opaque" }
		LOD 200

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows vertex:vert finalcolor:clearCoat

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		#include "HLSLSupport.cginc"
		#include "UnityShaderVariables.cginc"
		#include "UnityShaderUtilities.cginc"
		#include "UnityCG.cginc"
		#include "Lighting.cginc"
		#include "UnityPBSLighting.cginc"
		#include "AutoLight.cginc"

		sampler2D _MainTex;
		sampler2D _NormalMap;
		sampler2D _Metallic;

		struct Input {
			float2 uv_MainTex;
			float2 uv_NormalMap;
			float3 viewDir;
			float3 worldPos;
			float3 worldNormal ;// INTERNAL_DATA 
			float3 originalNormal;
		};

		fixed4 _Color;
		float _ClearCoat_Smoothness;
		float _ClearCoat_Metallic;

		// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
		// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
		// #pragma instancing_options assumeuniformscaling
		UNITY_INSTANCING_BUFFER_START(Props)
			// put more per-instance properties here
		UNITY_INSTANCING_BUFFER_END(Props)

		void vert (inout appdata_full v, out Input o) {
	        UNITY_INITIALIZE_OUTPUT(Input,o);
	        o.originalNormal = UnityObjectToWorldNormal(v.normal);
	    }

	    half4 UNITY_BRDF_PBS_ClearCoat (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
		    float3 normal, float3 viewDir,
		    UnityLight light, UnityIndirect gi)
		{
		    float perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
		    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);

		// NdotV should not be negative for visible pixels, but it can happen due to perspective projection and normal mapping
		// In this case normal should be modified to become valid (i.e facing camera) and not cause weird artifacts.
		// but this operation adds few ALU and users may not want it. Alternative is to simply take the abs of NdotV (less correct but works too).
		// Following define allow to control this. Set it to 0 if ALU is critical on your platform.
		// This correction is interesting for GGX with SmithJoint visibility function because artifacts are more visible in this case due to highlight edge of rough surface
		// Edit: Disable this code by default for now as it is not compatible with two sided lighting used in SpeedTree.
		#define UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV 0

		#if UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV
		    // The amount we shift the normal toward the view vector is defined by the dot product.
		    half shiftAmount = dot(normal, viewDir);
		    normal = shiftAmount < 0.0f ? normal + viewDir * (-shiftAmount + 1e-5f) : normal;
		    // A re-normalization should be applied here but as the shift is small we don't do it to save ALU.
		    //normal = normalize(normal);

		    half nv = saturate(dot(normal, viewDir)); // TODO: this saturate should no be necessary here
		#else
		    half nv = abs(dot(normal, viewDir));    // This abs allow to limit artifact
		#endif

		    half nl = saturate(dot(normal, light.dir));
		    float nh = saturate(dot(normal, halfDir));

		    half lv = saturate(dot(light.dir, viewDir));
		    half lh = saturate(dot(light.dir, halfDir));

		    // Diffuse term
		    half diffuseTerm = DisneyDiffuse(nv, nl, lh, perceptualRoughness) * nl;

		    // Specular term
		    // HACK: theoretically we should divide diffuseTerm by Pi and not multiply specularTerm!
		    // BUT 1) that will make shader look significantly darker than Legacy ones
		    // and 2) on engine side "Non-important" lights have to be divided by Pi too in cases when they are injected into ambient SH
		    float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
		#if UNITY_BRDF_GGX
		    // GGX with roughtness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughtness remapping.
		    roughness = max(roughness, 0.002);
		    half V = SmithJointGGXVisibilityTerm (nl, nv, roughness);
		    float D = GGXTerm (nh, roughness);
		#else
		    // Legacy
		    half V = SmithBeckmannVisibilityTerm (nl, nv, roughness);
		    half D = NDFBlinnPhongNormalizedTerm (nh, PerceptualRoughnessToSpecPower(perceptualRoughness));
		#endif

		    half specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later

		#   ifdef UNITY_COLORSPACE_GAMMA
		        specularTerm = sqrt(max(1e-4h, specularTerm));
		#   endif

		    // specularTerm * nl can be NaN on Metal in some cases, use max() to make sure it's a sane value
		    specularTerm = max(0, specularTerm * nl);
		#if defined(_SPECULARHIGHLIGHTS_OFF)
		    specularTerm = 0.0;
		#endif

		    // surfaceReduction = Int D(NdotH) * NdotH * Id(NdotL>0) dH = 1/(roughness^2+1)
		    half surfaceReduction;
		#   ifdef UNITY_COLORSPACE_GAMMA
		        surfaceReduction = 1.0-0.28*roughness*perceptualRoughness;      // 1-0.28*x^3 as approximation for (1/(x^4+1))^(1/2.2) on the domain [0;1]
		#   else
		        surfaceReduction = 1.0 / (roughness*roughness + 1.0);           // fade \in [0.5;1]
		#   endif

		    // To provide true Lambert lighting, we need to be able to kill specular completely.
		    specularTerm *= any(specColor) ? 1.0 : 0.0;

		    half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));
		    half3 color =  specularTerm * light.color * FresnelTerm (specColor, lh)
		                    + surfaceReduction * gi.specular * FresnelLerp (specColor, grazingTerm, nv);
		    //color = gi.specular;
		    //return specularTerm;
		    return half4(color, 1);
		}

	    inline half4 Standard_ClearCoat (SurfaceOutputStandard s, float3 viewDir, UnityGI gi)
		{
		    s.Normal = normalize(s.Normal);

		    half oneMinusReflectivity;
		    half3 specColor;
		    s.Albedo = DiffuseAndSpecularFromMetallic (s.Albedo, s.Metallic, /*out*/ specColor, /*out*/ oneMinusReflectivity);

		    // shader relies on pre-multiply alpha-blend (_SrcBlend = One, _DstBlend = OneMinusSrcAlpha)
		    // this is necessary to handle transparency in physically correct way - only diffuse component gets affected by alpha
		    half outputAlpha;
		    s.Albedo = PreMultiplyAlpha (s.Albedo, s.Alpha, oneMinusReflectivity, /*out*/ outputAlpha);

		    half4 c = UNITY_BRDF_PBS_ClearCoat (s.Albedo, specColor, oneMinusReflectivity, s.Smoothness, s.Normal, viewDir, gi.light, gi.indirect);
		    c.a = outputAlpha;
		    return c;
		}
		
		void clearCoat(Input IN, SurfaceOutputStandard o, inout fixed4 color)
		{

			  #ifndef USING_DIRECTIONAL_LIGHT
			    fixed3 lightDir = normalize(UnityWorldSpaceLightDir(IN.worldPos));
			  #else
			    fixed3 lightDir = _WorldSpaceLightPos0.xyz;
			  #endif
			  IN.viewDir = normalize(UnityWorldSpaceViewDir(IN.worldPos));
			  UnityGI gi;
			  UNITY_INITIALIZE_OUTPUT(UnityGI, gi);
			  gi.indirect.diffuse = 0;
			  gi.indirect.specular = 0;
			  gi.light.color = _LightColor0.rgb;
			  gi.light.dir = lightDir;
			  gi.light.ndotl = 1;
			  // Call GI (lightmaps/SH/reflections) lighting function
			  UnityGIInput giInput = (UnityGIInput)0;
			  //UNITY_INITIALIZE_OUTPUT(UnityGIInput, giInput);
			  giInput.light = gi.light;
			  giInput.worldPos = IN.worldPos;
			  giInput.worldViewDir = IN.viewDir;
			  giInput.probeHDR[0] = unity_SpecCube0_HDR;
			  giInput.probeHDR[1] = unity_SpecCube1_HDR;
			  #if defined(UNITY_SPECCUBE_BLENDING) || defined(UNITY_SPECCUBE_BOX_PROJECTION)
			    giInput.boxMin[0] = unity_SpecCube0_BoxMin; // .w holds lerp value for blending
			  #endif
			  o.Normal = IN.originalNormal;
			  o.Smoothness = _ClearCoat_Smoothness;
			  o.Metallic = _ClearCoat_Metallic;
			  LightingStandard_GI(o, giInput, gi);
			  color +=  Standard_ClearCoat (o, IN.viewDir, gi);

		}	

		void surf (Input IN, inout SurfaceOutputStandard o) {
			// Albedo comes from a texture tinted by color
			fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
			fixed4 meta = tex2D(_Metallic, IN.uv_MainTex);
			o.Albedo = c.rgb;
			// Metallic and smoothness come from slider variables
			o.Metallic = meta.r;
			o.Smoothness = meta.a;
			o.Normal = UnpackNormal(tex2D(_NormalMap, IN.uv_NormalMap));
			o.Alpha = c.a;

		}
		ENDCG
	}
	FallBack "Diffuse"
}
